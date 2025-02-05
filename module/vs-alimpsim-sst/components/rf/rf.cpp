#include "rf.h"

#include "activationEvent.h"
#include "instructionEvent.h"
#include "memoryEvents.h"

RegisterFile::RegisterFile(ComponentId_t id, Params &params)
    : DRRAComponent(id, params) {
  // Clock
  TimeConverter *tc = registerClock(
      clock,
      new SST::Clock::Handler<RegisterFile>(this, &RegisterFile::clockTick));

  // Controller port
  controller_link = configureLink(
      "controller_port", "0ns",
      new Event::Handler<RegisterFile>(this, &RegisterFile::handleEvent));

  // Data port (switchbox connection)
  data_link = configureLink("data_port", new Event::Handler<RegisterFile>(
                                             this, &RegisterFile::handleEvent));

  // Register file parameters
  access_time = params.find<std::string>("access_time", "0ns");
  register_file_size = params.find<int>("register_file_size", 1024);
}

RegisterFile::~RegisterFile() {}

void RegisterFile::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void RegisterFile::setup() {
  out.verbose(CALL_INFO, 1, 0, "Setup\n");
  if (slot_id == 1) {
    // Initialize registers
    // out.output("Initializing register file\n");
    for (int i = 0; i < register_file_size; i++) {
      registers[i] = 10 + i;
      // out.output("registers[%d] = %d\n", i, registers[i]);
    }
  }
}

void RegisterFile::complete(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Completed\n");
}

void RegisterFile::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool RegisterFile::clockTick(Cycle_t currentCycle) {
  if (currentCycle % printFrequency == 0) {
    out.output("--- REGISTERFILE CYCLE %" PRIu64 " ---\n", currentCycle);
  }

  if (isActive) {
    auto events = timingState.getEventsForCycle(activeCycle);
    out.output("Active cycle %lu: %lu events\n", activeCycle, events.size());
    for (auto event : events) {
      eventsHandlers[event->getEventNumber()](); // Call event handler
    }

    if (current_dsu_mode == -1) {
      out.fatal(CALL_INFO, -1, "DSU mode not set\n");
    }
    activeCycle++;
  }
  return false;
}

void RegisterFile::handleEvent(Event *event) {
  if (event) {
    // Check if the event is an ActEvent
    ActEvent *actEvent = dynamic_cast<ActEvent *>(event);
    if (actEvent) {
      out.output("Received ActEvent\n");
      isActive = true;
      timingState.build();
      out.output("Built timing model: %s\n", timingState.toString().c_str());
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

    // Check if the event is a memory request
    MemoryEvent *readReq = dynamic_cast<MemoryEvent *>(event);
    if (readReq) {
      out.output("Received memory request\n");
      return;
    }

    // Check if the event is a data event
    DataEvent *dataEvent = dynamic_cast<DataEvent *>(event);
    if (dataEvent) {
      switch (dataEvent->portType) {
      case DataEvent::PortType::WriteNarrow:
        if (current_dsu_mode != DataEvent::PortType::ReadNarrow) {
          out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
        } else {
          out.output("Received data event of size %d bytes (%d words): [",
                     dataEvent->size / 8, dataEvent->size / word_bitwidth);
          for (int i = 0; i < (dataEvent->size / 8) - 1; i++) {
            out.print("%d: %d, ", i, dataEvent->payload[i]);
          }
          out.print("%d: %d]\n", (dataEvent->size / 8) - 1,
                    dataEvent->payload[(dataEvent->size / 8) - 1]);
        }
        break;
      case DataEvent::PortType::ReadNarrow:
        if (current_dsu_mode != DataEvent::PortType::WriteNarrow) {
          out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
        }
        break;
      case DataEvent::PortType::WriteWide:
        if (current_dsu_mode != DataEvent::PortType::ReadWide) {
          out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
        }
        break;
      case DataEvent::PortType::ReadWide:
        if (current_dsu_mode != DataEvent::PortType::WriteWide) {
          out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
        }
        break;

      default:
        out.fatal(CALL_INFO, -1, "Invalid port type\n");
        break;
      }
      out.output("Received data event\n");
      return;
    }
  }
}

void RegisterFile::decodeInstr(uint32_t instr) {
  // Decode instruction
  uint32_t opcode = getInstrOpcode(instr);
  switch (opcode) {
  case REP: // repetition instruction
    handleRep(instr);
    break;
  case REPX: // repetition instruction
    handleRepx(instr);
    break;
  case DSU: // data setup instruction
    handleDSU(instr);
    break;
  default:
    out.fatal(CALL_INFO, -1, "Invalid opcode\n");
  }
}

void RegisterFile::handleRep(uint32_t instr) {
  // Instruction fields
  uint32_t port = getInstrField(instr, 2, 22);
  uint32_t level = getInstrField(instr, 4, 18);
  uint32_t iter = getInstrField(instr, 6, 12);
  uint32_t step = getInstrField(instr, 6, 6);
  uint32_t delay = getInstrField(instr, 6, 0);

  // For now, we only support increasing repetition levels (and no skipping)
  if (level != lastRepLevel + 1) {
    out.fatal(CALL_INFO, -1, "Invalid repetition level (last=%u, curr=%u)\n",
              lastRepLevel, level);
  } else {
    lastRepLevel = level;
  }

  // add repetition to the timing model
  try {
    timingState.addRepetition(iter, step);
  } catch (const std::exception &e) {
    out.fatal(CALL_INFO, -1, "Failed to add repetition: %s\n", e.what());
  }
}

void RegisterFile::handleRepx(uint32_t instr) { handleRep(instr); }

void RegisterFile::handleDSU(uint32_t instr) {
  // Instruction fields
  bool init_addr_sd = getInstrField(instr, 1, 23) == 1;
  uint16_t init_addr = getInstrField(instr, 16, 7);
  uint32_t port = getInstrField(instr, 2, 5);

  uint32_t dsu_mode_to_set = port;
  timingState.addEvent("dsu_" + std::to_string(current_event_number));

  // Set initial address
  agu_initial_addr = init_addr;
  current_dsu_mode = dsu_mode_to_set;

  // Add the event handler
  switch (current_dsu_mode) {
  case DataEvent::PortType::WriteNarrow:
    eventsHandlers.push_back([this] { sendNarrowData(); });
    break;
  case DataEvent::PortType::ReadNarrow:
    eventsHandlers.push_back([this] { receiveNarrowData(); });
    break;
  case DataEvent::PortType::WriteWide:
    eventsHandlers.push_back([this] { sendWideData(); });
    break;
  case DataEvent::PortType::ReadWide:
    eventsHandlers.push_back([this] { receiveWideData(); });
    break;

  default:
    out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
  }

  // Add event handler
  current_event_number++;
}

void RegisterFile::sendWideData() {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteWide);
  vector<uint8_t> data;
  out.output("Sending data of size %d bytes (%d words): [", io_data_width / 8,
             io_data_width / word_bitwidth);
  for (int i = 0; i < io_data_width / 8; i++) {
    data.push_back(registers[agu_initial_addr + i]);
    out.print("%d: %d, ", i, data[i]);
  }
  out.print("]\n");
  dataEvent->size = io_data_width;
  dataEvent->payload = data;
  data_link->send(dataEvent);
  agu_initial_addr += io_data_width / 8;
}

void RegisterFile::sendNarrowData() {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  vector<uint8_t> data = {};
  for (int i = 0; i < word_bitwidth / 8; i++) {
    data.push_back((registers[agu_initial_addr] >> (i * 8)) & 0xFF);
  }
  dataEvent->size = word_bitwidth;
  dataEvent->payload = data;
  out.output("Sending data of size %d bytes (%d words): [", dataEvent->size / 8,
             dataEvent->size / word_bitwidth);
  for (int i = 0; i < (dataEvent->size / 8) - 1; i++) {
    out.print("%d: %d, ", i, dataEvent->payload[i]);
  }
  out.print("%d: %d]\n", (dataEvent->size / 8) - 1,
            dataEvent->payload[(dataEvent->size / 8) - 1]);
  data_link->send(dataEvent);
  agu_initial_addr++;
}

void RegisterFile::receiveWideData() {
  //   out.output("Receive wide data\n");
  //   // Receive data from the switchbox
  //   DataEvent *dataEvent;

  //   while ((dataEvent = dynamic_cast<DataEvent *>(data_link->recv())) !=
  //          nullptr) {
  //     if (dataEvent->portType != DataEvent::PortType::WriteWide) {
  //       out.fatal(CALL_INFO, -1, "Invalid port type\n");
  //     }
  //     for (int i = 0; i < io_data_width / 8; i++) {
  //       registers[agu_initial_addr + i] = dataEvent->payload[i];
  //     }

  //     out.output("dataEvent->size = %d\n", dataEvent->size);
  //     out.output("dataEvent->payload = [");
  //     for (int i = 0; i < (dataEvent->size / 8) - 1; i++) {
  //       out.print("%d: %d, ", i, dataEvent->payload[i]);
  //     }
  //     out.print("%d: %d]\n", (dataEvent->size / 8) - 1,
  //               dataEvent->payload[(dataEvent->size / 8) - 1]);
  //   }
}

void RegisterFile::receiveNarrowData() {
  //   out.output("Waiting to receive narrow data...\n");
  //   // Receive data from the switchbox
  //   Event *event = data_link->recv();

  //   if (event == nullptr) {
  //     out.fatal(CALL_INFO, -1, "Failed to receive data event\n");
  //   }
  //   DataEvent *dataEvent = dynamic_cast<DataEvent *>(event);

  //   if (dataEvent->portType != DataEvent::PortType::WriteNarrow) {
  //     out.fatal(CALL_INFO, -1, "Invalid port type\n");
  //   }
  //   uint64_t data = 0;
  //   for (int i = 0; i < word_bitwidth / 8; i++) {
  //     data |= dataEvent->payload[i] << (i * 8);
  //   }
  //   registers[agu_initial_addr] = data;

  //   out.output("dataEvent->size = %d\n", dataEvent->size);
  //   out.output("dataEvent->payload = [");
  //   for (int i = 0; i < (dataEvent->size / 8) - 1; i++) {
  //     out.print("%d: %d, ", i, dataEvent->payload[i]);
  //   }
  //   out.print("%d: %d]\n", (dataEvent->size / 8) - 1,
  //             dataEvent->payload[(dataEvent->size / 8) - 1]);
}