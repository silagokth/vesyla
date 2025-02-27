#include "io_sram.h"

#include "dataEvent.h"
#include "ioEvents.h"
#include "sst/core/event.h"
#include <bitset>

using namespace SST;

IOSRAM::IOSRAM(SST::ComponentId_t id, SST::Params &params)
    : DRRAResource(id, params) {
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

  // IO port
  io_link = configureLink("io_port", access_time);
}

void IOSRAM::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void IOSRAM::setup() {}

void IOSRAM::complete(unsigned int phase) {}

void IOSRAM::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool IOSRAM::clockTick(SST::Cycle_t currentCycle) {
  executeScheduledEventsForCycle(currentCycle);
  return false;
}

void IOSRAM::decodeInstr(uint32_t instr) {
  uint32_t instrType = getInstrType(instr);
  uint32_t instrOpcode = getInstrOpcode(instr);
  uint32_t instrSlot = getInstrSlot(instr);

  if (std::find(slot_ids.begin(), slot_ids.end(), instrSlot) ==
      slot_ids.end()) {
    out.fatal(CALL_INFO, -1, "Invalid slot: %u\n", instrSlot);
  }

  switch (instrOpcode) {
  case REP:
    handleRep(instr);
    break;
  case REPX:
    handleRepx(instr);
    break;
  case DSU:
    handleDSU(instr);
    break;
  default:
    out.fatal(CALL_INFO, -1, "Invalid opcode: %u\n", instrOpcode);
  }
}

void IOSRAM::handleRep(uint32_t instr) {
  // Instruction fields
  uint32_t slot = getInstrSlot(instr);
  uint32_t port = getInstrField(instr, 2, 22);
  uint32_t level = getInstrField(instr, 4, 18);
  uint32_t iter = getInstrField(instr, 6, 12);
  uint32_t step = getInstrField(instr, 6, 6);
  uint32_t delay = getInstrField(instr, 6, 0);

  out.output("rep (slot=%d, port=%d, level=%d, iter=%d, step=%d, delay=%d)\n",
             slot, port, level, iter, step, delay);

  uint32_t port_num = 0;
  auto it = std::find(slot_ids.begin(), slot_ids.end(), slot);
  if (it != slot_ids.end()) {
    port_num = std::distance(slot_ids.begin(), it);
  } else {
    out.fatal(CALL_INFO, -1, "Slot ID not found\n");
  }
  port_num = port_num * 4 + port;

  // For now, we only support increasing repetition levels (and no skipping)
  if (level != port_last_rep_level[port_num] + 1) {
    out.output("port_num = %d\n", port_num);
    out.fatal(
        CALL_INFO, -1,
        "Invalid repetition level (last=%d, curr=%d), instruction: %s (slot: "
        "%d, port: %d, level: %d, iter: %d, step: %d, delay: %d)\n",
        port_last_rep_level[port_num], level,
        std::bitset<32>(instr).to_string().c_str(), slot, port, level, iter,
        step, delay);
  } else {
    port_last_rep_level[port_num] = level;
  }

  // add repetition to the timing model
  try {
    next_timing_states[port_num].addRepetition(iter, delay, level, step);
    out.output("Added repetition to port %d (iter=%d, step=%d)\n", port_num,
               iter, step);
  } catch (const std::exception &e) {
    out.fatal(CALL_INFO, -1, "Failed to add repetition: %s\n", e.what());
  }
}

void IOSRAM::handleRepx(uint32_t instr) { handleRep(instr); }

void IOSRAM::handleDSU(uint32_t instr) {
  // Instruction fields
  uint32_t slot = getInstrSlot(instr);
  bool init_addr_sd = getInstrField(instr, 1, 23) == 1;
  uint16_t init_addr = getInstrField(instr, 16, 7);
  uint32_t port = getInstrField(instr, 2, 5);
  uint32_t port_num = getRelativePortNum(slot, port);

  out.output("dsu (slot: %d, init_addr_sd: %d, init_addr: %d, port: %d)\n",
             slot, init_addr_sd, init_addr, port);

  // Set initial address
  agu_initial_addr = init_addr;

  // Add the event handler
  switch (port_num) {
  case PortMap::SRAMReadFromIO:
    sram_read_from_io_initial_addr = init_addr;
    readFromIO();
    break;
  case PortMap::SRAMWriteToIO:
    sram_write_to_io_initial_addr = init_addr;
    writeToIO();
    break;
  case PortMap::IOWriteToSRAM:
    io_write_to_sram_initial_addr = init_addr;
    writeToSRAM();
    break;
  case PortMap::IOReadFromSRAM:
    io_read_from_sram_initial_addr = init_addr;
    readFromSRAM();
    break;
  case PortMap::WriteBulk:
    write_bulk_initial_addr = init_addr;
    writeBulk();
    break;
  case PortMap::ReadBulk:
    read_bulk_initial_addr = init_addr;
    readBulk();
    break;

  default:
    out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
  }

  // Add event handler
  current_event_number++;
}

void IOSRAM::readFromIO() {
  // Reading data from the IO to the buffer
  next_timing_states[PortMap::SRAMReadFromIO].addEvent(
      "dsu_read_from_io_" + std::to_string(current_event_number), 1, [this] {
        sram_read_from_io_address_buffer =
            sram_read_from_io_initial_addr +
            current_timing_states[PortMap::SRAMReadFromIO]
                .getRepIncrementForCycle(
                    getPortActiveCycle(PortMap::SRAMReadFromIO));

        IOReadRequest *readReq = new IOReadRequest();
        readReq->address = sram_read_from_io_address_buffer;
        readReq->size = io_data_width / 8;
        readReq->column_id = cell_coordinates[1];

        out.output("Sending read request to IO (addr=%d, size=%dbits)\n",
                   sram_read_from_io_address_buffer, io_data_width);

        io_link->send(readReq);
      });
}

void IOSRAM::writeToIO() {
  // Writing buffer data to the IO
  next_timing_states[PortMap::SRAMWriteToIO].addEvent(
      "dsu_write_to_io_" + std::to_string(current_event_number), 9, [this] {
        sram_write_to_io_address_buffer =
            sram_write_to_io_initial_addr +
            current_timing_states[PortMap::SRAMWriteToIO]
                .getRepIncrementForCycle(
                    getPortActiveCycle(PortMap::SRAMWriteToIO));

        IOWriteRequest *writeReq = new IOWriteRequest();
        writeReq->address = sram_write_to_io_address_buffer;
        writeReq->data = to_io_data_buffer;
        io_link->send(writeReq);

        out.output(
            "Sending write request to IO (addr=%d, size=%dbits, data=%s)\n",
            writeReq->address, writeReq->data.size() * 8,
            formatRawDataToWords(writeReq->data).c_str());
      });
}

void IOSRAM::writeToSRAM() {
  // Writing buffer data to the backend
  next_timing_states[PortMap::IOWriteToSRAM].addEvent(
      "dsu_write_to_sram_" + std::to_string(current_event_number), 8, [this] {
        // Check if the IO responded
        IOReadResponse *ioReadResponse =
            dynamic_cast<IOReadResponse *>(io_link->recv());
        if (ioReadResponse) {
          out.output("Received read response from IO (addr=%d, size=%dbits, "
                     "data=%s)\n",
                     ioReadResponse->address, ioReadResponse->data.size() * 8,
                     formatRawDataToWords(ioReadResponse->data).c_str());
          from_io_data_buffer = ioReadResponse->data;
          if (from_io_data_buffer.size() == 0) {
            out.fatal(CALL_INFO, -1, "No data from IO\n");
          }
        } else {
          out.fatal(CALL_INFO, -1, "No response from IO\n");
        }

        // Calculate the SRAM address
        io_write_to_sram_address_buffer =
            io_write_to_sram_initial_addr +
            current_timing_states[PortMap::IOWriteToSRAM]
                .getRepIncrementForCycle(
                    getPortActiveCycle(PortMap::IOWriteToSRAM));

        // Write data to the backend (SRAM)
        backend->set(io_write_to_sram_address_buffer,
                     from_io_data_buffer.size(), from_io_data_buffer);
        out.output("Writing to SRAM (adrr=%d, size=%dbits, data=%s)\n",
                   io_write_to_sram_address_buffer,
                   from_io_data_buffer.size() * 8,
                   formatRawDataToWords(from_io_data_buffer).c_str());

        // Clear the buffer
        from_io_data_buffer.clear();
      });
}

void IOSRAM::readFromSRAM() {
  // Reading data from the backend to the buffer
  next_timing_states[PortMap::IOReadFromSRAM].addEvent(
      "dsu_read_from_sram_" + std::to_string(current_event_number), 2, [this] {
        io_read_from_sram_address_buffer =
            io_read_from_sram_initial_addr +
            current_timing_states[PortMap::IOReadFromSRAM]
                .getRepIncrementForCycle(
                    getPortActiveCycle(PortMap::IOReadFromSRAM));

        to_io_data_buffer.resize(io_data_width / 8);
        for (int i = 0; i < io_data_width / 8; i++) {
          to_io_data_buffer[i] = 0;
        }

        backend->get(io_read_from_sram_address_buffer, io_data_width / 8,
                     to_io_data_buffer);

        out.output("Reading from SRAM (addr=%d, size=%dbits, data=%s)\n",
                   io_read_from_sram_address_buffer, io_data_width,
                   formatRawDataToWords(to_io_data_buffer).c_str());
      });
}

void IOSRAM::readBulk() {
  out.output("Add event read bulk (port %d prio %d)\n", PortMap::ReadBulk, 1);
  next_timing_states[PortMap::ReadBulk].addEvent(
      "dsu_send_" + std::to_string(current_event_number), 1, [this] {
        read_bulk_address_buffer =
            read_bulk_initial_addr +
            current_timing_states[PortMap::ReadBulk].getRepIncrementForCycle(
                getPortActiveCycle(PortMap::ReadBulk));

        DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteWide);
        vector<uint8_t> data;
        data.resize(io_data_width / 8);
        backend->get(read_bulk_address_buffer, io_data_width / 8, data);
        out.output("Reading bulk data (addr=%d, size=%dbits, data=%s)\n",
                   read_bulk_address_buffer, io_data_width,
                   formatRawDataToWords(data).c_str());
        dataEvent->size = io_data_width;
        dataEvent->payload = data;
        data_links[1]->send(dataEvent);
      });
}

void IOSRAM::writeBulk() {
  next_timing_states[PortMap::WriteBulk].addEvent(
      "dsu_receive_" + std::to_string(current_event_number), 9, [this] {
        write_bulk_address_buffer =
            write_bulk_initial_addr +
            current_timing_states[PortMap::WriteBulk].getRepIncrementForCycle(
                getPortActiveCycle(PortMap::WriteBulk));

        // Check if some data was received
        DataEvent *dataEvent = dynamic_cast<DataEvent *>(data_links[1]->recv());
        if (dataEvent == nullptr)
          out.fatal(CALL_INFO, -1, "No data received\n");

        out.output("Writing bulk data (addr=%d, size=%dbits, data=%s)\n",
                   write_bulk_address_buffer, dataEvent->size,
                   formatRawDataToWords(dataEvent->payload).c_str());

        // Write data to the backend
        backend->set(write_bulk_address_buffer, dataEvent->size,
                     dataEvent->payload);
      });
}
