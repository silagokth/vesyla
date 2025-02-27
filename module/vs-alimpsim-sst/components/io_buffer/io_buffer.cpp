#include "io_buffer.h"

#include <sst/core/componentExtension.h>
#include <sst/core/interfaces/stdMem.h>

#include "custom_backing.h"

#include "ioEvents.h"

IOBuffer::IOBuffer(SST::ComponentId_t id, SST::Params &params) : Component(id) {
  // Output
  out.init("", 0, 0, Output::STDOUT);
  out.setPrefix(getType() + " - ");

  // Get params
  clock = params.find<std::string>("clock", "100MHz");
  printFrequency = params.find<SST::Cycle_t>("printFrequency", 1000);
  io_data_width = params.find<uint32_t>("io_data_width", 256);
  io_depth = params.find<uint32_t>("io_depth", 65536);
  word_bitwidth = params.find<uint32_t>("word_bitwidth", 16);
  access_time = params.find<std::string>("access_time", "0ns");
  num_columns = params.find<uint32_t>("num_columns", 1);
  read_only = params.find<bool>("read_only", false);

  // Clock
  SST::TimeConverter *tc = registerClock(
      clock, new SST::Clock::Handler<IOBuffer>(this, &IOBuffer::clockTick));

  // Backing store
  bool found = false;
  std::string backingType = params.find<std::string>(
      "backing", "mfile",
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

  if (backingType == "mfile") {
    std::string memoryFile = params.find<std::string>("memory_file", "");
    if (0 == memoryFile.compare("")) {
      memoryFile.clear();
    }
    try {
      backend = new SST::MemHierarchy::Backend::BackingIO(
          memoryFile, io_data_width, io_depth, read_only);
    } catch (int e) {
      if (e == 1) {
        out.fatal(CALL_INFO, -1, "Failed to open memory file: %s\n",
                  memoryFile.c_str());
      } else {
        out.fatal(CALL_INFO, -1, "Failed to map memory file: %s\n",
                  memoryFile.c_str());
      }
    }

  } else if (backingType == "mmap") {
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
  column_links.reserve(num_columns);
  std::vector<uint32_t> connected_links;
  for (uint32_t i = 0; i < num_columns; i++) {
    SST::Link *link = configureLink(
        "col_port" + std::to_string(i), access_time,
        new SST::Event::Handler2<IOBuffer, &IOBuffer::handleEventFromColumn,
                                 uint32_t>(this, i));
    sst_assert(link, CALL_INFO, -1, "Failed to configure column link %d\n", i);

    column_links[i] = link;
    connected_links.push_back(i);
  }
  out.output("Connected %lu column links (", connected_links.size());
  for (auto link : connected_links) {
    out.print("%u,", link);
  }
  out.print(")\n");
}

void IOBuffer::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "IO Buffer Initialized\n");
}

void IOBuffer::setup() {}

void IOBuffer::complete(unsigned int phase) {}

void IOBuffer::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool IOBuffer::clockTick(SST::Cycle_t currentCycle) {
  if (currentCycle % 10 == 0) {
    out.output("--- CYCLE %" PRIu64 " ---\n", currentCycle / 10);
  }

  return false;
}

void IOBuffer::handleEventFromColumn(SST::Event *event, uint32_t column_id) {
  IOReadRequest *readReq = dynamic_cast<IOReadRequest *>(event);
  std::vector<uint8_t> data;
  if (readReq) {
    out.output("Received read request (addr=%d, size=%dbits)\n",
               readReq->address, readReq->size * 8);
    data.reserve(readReq->size);
    if (backend) {
      backend->get(readReq->address, readReq->size * 8 / io_data_width, data);
    } else {
      data.resize(readReq->size, 0); // send zeros if no backend
    }

    read_address_buffer = readReq->address;
    read_data_buffer = data;

    IOReadResponse *readResp = new IOReadResponse();
    readResp->address = readReq->address;
    readResp->set_data(data);
    column_links[column_id]->send(readResp);
    out.output("Reading from IOBuffer (addr=%d, size=%dbits, data=%s)\n",
               readReq->address, readReq->size * 8,
               formatRawDataToWords(data).c_str());

    return;
  }

  IOWriteRequest *writeReq = dynamic_cast<IOWriteRequest *>(event);
  if (writeReq) {
    out.output("Received write request (addr=%d, size=%dbits)\n",
               writeReq->address, writeReq->data.size() * 8);
    if (backend) {
      write_address_buffer = writeReq->address;
      write_data_buffer = writeReq->data;
      out.output("Writing to IOBuffer (addr=%d, size=%dbits, data=%s)\n",
                 writeReq->address, writeReq->data.size() * 8,
                 formatRawDataToWords(writeReq->data).c_str());

      backend->set(writeReq->address, writeReq->data.size(), writeReq->data);
    }

    return;
  }
}