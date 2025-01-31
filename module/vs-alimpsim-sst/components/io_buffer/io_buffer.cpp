#include "io_buffer.h"

#include <sst/core/componentExtension.h>
#include <sst/core/interfaces/stdMem.h>

#include "memoryEvents.h"

IOBuffer::IOBuffer(SST::ComponentId_t id, SST::Params &params)
    : DRRAComponent(id, params) {
  // Clock
  SST::TimeConverter *tc = registerClock(
      clock, new SST::Clock::Handler<IOBuffer>(this, &IOBuffer::clockTick));

  access_time = params.find<std::string>("access_time", "0ns");

  // Backing store
  bool found = false;
  std::string backingType = params.find<std::string>(
      "backing", "malloc",
      found); /* Default to using an mmap backing store, fall back on malloc */
  if (!found) {
    bool oldBackVal = params.find<bool>("do-not-back", false, found);
    if (oldBackVal)
      backingType = "none";
  }

  // Backend
  std::string mallocSize =
      params.find<std::string>("backing_size_unit", "1MiB");
  SST::UnitAlgebra size(mallocSize);
  if (!size.hasUnits("B")) {
    out.fatal(CALL_INFO, -1, "Invalid memory size specified: %s\n",
              mallocSize.c_str());
  }
  size_t sizeBytes = size.getRoundedValue();

  if (backingType == "mmap") {
    std::string memoryFile = params.find<std::string>("memory_file", "");
    if (0 == memoryFile.compare("")) {
      memoryFile.clear();
    }
    try {
      backend =
          new SST::MemHierarchy::Backend::BackingMMAP(memoryFile, sizeBytes);
    } catch (int e) {
      if (e == 1) {
        out.fatal(CALL_INFO, -1, "Failed to open memory file: %s\n",
                  memoryFile.c_str());
      } else {
        out.fatal(CALL_INFO, -1, "Failed to map memory file: %s\n",
                  memoryFile.c_str());
      }
    }
  } else if (backingType == "malloc") {
    backend = new SST::MemHierarchy::Backend::BackingMalloc(sizeBytes);
  }
  out.output("Created backing store (type: %s)\n", backingType.c_str());

  // Column ports
  std::string linkPrefix = "col_port";
  std::string linkName = linkPrefix + "0";
  int portNum = 0;
  while (isPortConnected(linkName)) {
    SST::Link *link = configureLink(
        linkName, access_time,
        new SST::Event::Handler<IOBuffer>(this, &IOBuffer::handleEvent));
    sst_assert(link, CALL_INFO, -1, "Failed to configure link %s\n",
               linkName.c_str());

    if (!link) {
      out.fatal(CALL_INFO, -1, "Failed to configure link %s\n",
                linkName.c_str());
    }
    columnLinks.push_back(link);

    // Next link
    portNum++;
    linkName = linkPrefix + std::to_string(portNum);
  }
}

void IOBuffer::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void IOBuffer::setup() { out.verbose(CALL_INFO, 1, 0, "Setup\n"); }

void IOBuffer::complete(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Completed\n");
}

void IOBuffer::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool IOBuffer::clockTick(SST::Cycle_t currentCycle) {
  if (currentCycle % printFrequency == 0) {
    out.output("--- IOBUFFER CYCLE %" PRIu64 " ---\n", currentCycle);
  }
  return true;
}

void IOBuffer::handleEvent(SST::Event *event) {
  ReadRequest *readReq = dynamic_cast<ReadRequest *>(event);
  std::vector<uint8_t> data;
  if (readReq) {
    out.output("Received read request\n");
    if (backend) {
      out.output("Reading from backend\n");
      for (int i = 0; i < readReq->size; i++) {
        data.push_back(backend->get(readReq->address + i));
      }
      out.output("Read from backend\n");
    } else {
      data.resize(readReq->size, 0); // Ensure proper size and alignment
    }

    ReadResponse *readResp = new ReadResponse();
    readResp->address = readReq->address;
    readResp->set_data(data);
    columnLinks[0]->send(readResp);
    out.output("Sent read response\n");

    return;
  }

  WriteRequest *writeReq = dynamic_cast<WriteRequest *>(event);
  if (writeReq) {
    out.output("Received write request\n");
    if (backend) {
      out.output("Writing to backend\n");
      backend->set(writeReq->address, writeReq->data.size(), writeReq->data);
      out.output("Data (%lu bits) = ", writeReq->data.size() * 8);
      for (int i = writeReq->data.size() - 1; i >= 0; i--) {
        out.print("%08b", writeReq->data[i]);
        if ((i % 2 == 0) && (i != 0)) {
          out.print(" ");
        }
      }
      out.print("\n");
      out.output("Wrote to backend\n");
    }

    return;
  }
}