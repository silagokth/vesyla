#include "io_sram.h"

#include "activationEvent.h"
#include "instructionEvent.h"
#include "ioEvents.h"
#include <bitset>

using namespace SST;

IOSRAM::IOSRAM(SST::ComponentId_t id, SST::Params &params)
    : DRRAResource(id, params) {
  // Clock
  SST::TimeConverter *tc = registerClock(
      clock, new SST::Clock::Handler<IOSRAM>(this, &IOSRAM::clockTick));

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
  io_link =
      configureLink("io_port", "0ns",
                    new Event::Handler<IOSRAM>(this, &IOSRAM::handleIOEvent));

  // // Controller port
  // controller_link =
  //     configureLink("controller_port", "0ns",
  //                   new Event::Handler<IOSRAM>(this, &IOSRAM::handleEvent));

  // // Data port
  // data_link =
  //     configureLink("data_port", "0ns",
  //                   new Event::Handler<IOSRAM>(this, &IOSRAM::handleEvent));
}

void IOSRAM::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void IOSRAM::setup() { out.verbose(CALL_INFO, 1, 0, "Setup\n"); }

void IOSRAM::complete(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Completed\n");
}

void IOSRAM::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool IOSRAM::clockTick(SST::Cycle_t currentCycle) {
  if (currentCycle % printFrequency == 0) {
    out.output("--- IOSRAM CYCLE %" PRIu64 " ---\n", currentCycle);
  }

  for (auto &port : active_ports) {
    if (isPortActive(port.first)) {
      auto events =
          getPortEventsForCycle(port.first, getPortActiveCycle(port.first));
      for (auto event : events) {
        event->execute();
      }
    }
  }

  return false;
}

void IOSRAM::handleIOEvent(Event *event) {
  IOEvent *ioEvent = dynamic_cast<IOEvent *>(event);
  if (ioEvent) {
    IOReadResponse *ioReadResponse = dynamic_cast<IOReadResponse *>(ioEvent);
    if (ioReadResponse) {
      // next_timing_states[PortMap::IOPortDSUIn].addEvent(
      //     "dsu_read_io_" + std::to_string(current_event_number),
      //     [this, ioReadResponse] {
      out.output("Received IO read response: address: %d, data: %s\n",
                 ioReadResponse->address,
                 formatRawDataToWords(ioReadResponse->data).c_str());
      // for (int i = 0; i < ioReadResponse->data.size(); i++) {
      //   if (i == ioReadResponse->data.size() - 1) {
      //     out.print("%d\n", ioReadResponse->data[i]);
      //     break;
      //   }
      //   out.print("%d, ", ioReadResponse->data[i]);
      // }
      // Write data to the backend
      backend->set(ioReadResponse->address, ioReadResponse->data.size(),
                   ioReadResponse->data);
      // });
      return;
    }
  }
}

void IOSRAM::handleEvent(SST::Event *event) {
  if (event) {
    // Check if the event is an ActEvent
    ActEvent *actEvent = dynamic_cast<ActEvent *>(event);
    if (actEvent) {
      out.output("Received ActEvent: slot_id: %d, ports: %s\n",
                 actEvent->slot_id,
                 bitset<4>(actEvent->ports).to_string().c_str());
      activatePortsForSlot(actEvent->slot_id, actEvent->ports);

      return;
    }

    // Check if the event is an InstrEvent
    InstrEvent *instrEvent = dynamic_cast<InstrEvent *>(event);
    if (instrEvent) {
      instrBuffer = instrEvent->instruction;
      decodeInstr(instrBuffer);
      out.output("Received InstrEvent: %08x\n", instrBuffer);
      return;
    }

    // Check if the event is a data event
    DataEvent *dataEvent = dynamic_cast<DataEvent *>(event);
    if (dataEvent) {
      switch (dataEvent->portType) {
      // case DataEvent::PortType::WriteNarrow:
      //   if (current_dsu_mode != DataEvent::PortType::ReadNarrow) {
      //     out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
      //   } else {
      //     out.output("Received data event of size %d bytes (%d words): [",
      //                dataEvent->size / 8, dataEvent->size / word_bitwidth);
      //     for (int i = 0; i < (dataEvent->size / 8) - 1; i++) {
      //       out.print("%d: %d, ", i, dataEvent->payload[i]);
      //     }
      //     out.print("%d: %d]\n", (dataEvent->size / 8) - 1,
      //               dataEvent->payload[(dataEvent->size / 8) - 1]);
      //   }
      //   break;
      // case DataEvent::PortType::ReadNarrow:
      //   if (current_dsu_mode != DataEvent::PortType::WriteNarrow) {
      //     out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
      //   }
      //   break;
      case DataEvent::PortType::WriteWide:
        receiveWideData(dataEvent);
        break;
        // case DataEvent::PortType::ReadWide:
        //   if (current_dsu_mode != DataEvent::PortType::WriteWide) {
        //     out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
        //   }
        //   break;

      default:
        out.fatal(CALL_INFO, -1, "Invalid port type\n");
        break;
      }
      out.output("Received data event\n");
      return;
    }
  }
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
    next_timing_states[port_num].addRepetition(iter, step);
  } catch (const std::exception &e) {
    out.fatal(CALL_INFO, -1, "Failed to add repetition: %s\n", e.what());
  }
}

void IOSRAM::handleRepx(uint32_t instr) { handleRep(instr); }

void IOSRAM::handleDSU(uint32_t instr) {
  // Instruction fields
  bool init_addr_sd = getInstrField(instr, 1, 23) == 1;
  uint16_t init_addr = getInstrField(instr, 16, 7);
  uint32_t port = getInstrField(instr, 2, 5);

  uint32_t dsu_mode_to_set = port;

  // Set initial address
  agu_initial_addr = init_addr;

  // Add the event handler
  switch (port) {
  case PortMap::IOPortDSUIn:
  case PortMap::IOPortAGUIn:
    readFromIO();
    break;
  case PortMap::IOPortDSUOut:
  case PortMap::IOPortAGUOut:
    writeToIO();
    break;
  case PortMap::BulkOut:
    sendWideData();
    break;
  case PortMap::BulkIn:
    out.output("Reading wide data\n");
    break;

  default:
    out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
  }

  // Add event handler
  current_event_number++;
}

void IOSRAM::readFromIO() {
  next_timing_states[PortMap::IOPortDSUIn].addEvent(
      "dsu_read_io_" + std::to_string(current_event_number), [this] {
        IOReadRequest *readReq = new IOReadRequest();
        readReq->address = agu_initial_addr;
        readReq->size = io_data_width / 8;
        readReq->column_id = cell_coordinates[1];

        out.output("Read request IO address: %d\n", agu_initial_addr);

        io_link->send(readReq);
      });
}

void IOSRAM::writeToIO() {
  next_timing_states[PortMap::IOPortDSUOut].addEvent(
      "dsu_write_io_" + std::to_string(current_event_number), [this] {
        IOWriteRequest *writeReq = new IOWriteRequest();
        writeReq->address = agu_initial_addr;
        for (int i = 0; i < io_data_width / 8; i++) {
          writeReq->data.push_back(backend->get(agu_initial_addr + i));
        }
        io_link->send(writeReq);
      });
}

void IOSRAM::sendWideData() {
  next_timing_states[PortMap::BulkOut].addEvent(
      "dsu_send_" + std::to_string(current_event_number), [this] {
        DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteWide);
        vector<uint8_t> data;
        out.output("Sending data of size %d bytes (%d words): [",
                   io_data_width / 8, io_data_width / word_bitwidth);
        for (int i = 0; i < io_data_width / 8; i++) {
          data.push_back(backend->get(agu_initial_addr + i));
          out.print("%d: %d, ", i, data[i]);
        }
        out.print("]\n");
        dataEvent->size = io_data_width;
        dataEvent->payload = data;
        data_links[0]->send(dataEvent);
        agu_initial_addr += io_data_width / 8;
      });
}

void IOSRAM::receiveWideData(DataEvent *dataEvent) {
  next_timing_states[PortMap::BulkIn].addEvent(
      "dsu_receive_" + std::to_string(current_event_number), [this, dataEvent] {
        out.output("Received data event of size %d bytes (%d words): [",
                   dataEvent->size / 8, dataEvent->size / word_bitwidth);

        // Write data to the backend
        backend->set(agu_initial_addr, dataEvent->size, dataEvent->payload);

        for (int i = 0; i < (dataEvent->size / 8) - 1; i++) {
          out.print("%d: %d, ", i, dataEvent->payload[i]);
        }
        out.print("%d: %d]\n", (dataEvent->size / 8) - 1,
                  dataEvent->payload[(dataEvent->size / 8) - 1]);
      });
}
