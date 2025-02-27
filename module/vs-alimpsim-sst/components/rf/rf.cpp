#include "rf.h"

#include "dataEvent.h"

RegisterFile::RegisterFile(ComponentId_t id, Params &params)
    : DRRAResource(id, params) {
  // Register file parameters
  access_time = params.find<std::string>("access_time", "0ns");
  register_file_size = params.find<int>("register_file_size", 64);
}

RegisterFile::~RegisterFile() {}

void RegisterFile::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void RegisterFile::setup() {
  for (int i = 0; i < register_file_size; i++) {
    for (int j = 0; j < word_bitwidth / 8; j++) {
      registers[i].push_back(0);
    }
  }
}

void RegisterFile::complete(unsigned int phase) {}

void RegisterFile::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool RegisterFile::clockTick(Cycle_t currentCycle) {
  executeScheduledEventsForCycle(currentCycle);
  return false;
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
    out.fatal(CALL_INFO, -1, "Invalid repetition level (last=%u, curr=%u)\n",
              port_last_rep_level[port_num], level);
  } else {
    port_last_rep_level[port_num] = level;
  }

  // add repetition to the timing model
  try {
    next_timing_states[port_num].addRepetition(iter, delay, level, step);
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

  port_agus_init[port] = init_addr;

  // Add the event handler
  switch (port) {
  case DataEvent::PortType::ReadNarrow:
    next_timing_states[port].addEvent(
        "dsu_read_narrow_" + std::to_string(current_event_number), 1, [this] {
          updatePortAGUs(DataEvent::PortType::ReadNarrow);
          readNarrow();
        });
    break;
  case DataEvent::PortType::ReadWide:
    next_timing_states[port].addEvent(
        "dsu_read_wide_" + std::to_string(current_event_number), 1, [this] {
          updatePortAGUs(DataEvent::PortType::ReadWide);
          readWide();
        });
    break;
  case DataEvent::PortType::WriteNarrow:
    next_timing_states[port].addEvent(
        "dsu_write_narrow_" + std::to_string(current_event_number), 9, [this] {
          updatePortAGUs(DataEvent::PortType::WriteNarrow);
          writeNarrow();
        });
    break;
  case DataEvent::PortType::WriteWide:
    next_timing_states[port].addEvent(
        "dsu_write_wide_" + std::to_string(current_event_number), 9, [this] {
          updatePortAGUs(DataEvent::PortType::WriteWide);
          writeWide();
        });
    break;

  default:
    out.fatal(CALL_INFO, -1, "Invalid DSU mode\n");
  }

  // Add event handler
  current_event_number++;
}

void RegisterFile::readWide() {
  vector<uint8_t> data;
  uint32_t addr =
      port_agus[DataEvent::PortType::ReadWide] * io_data_width / word_bitwidth;

  out.output("Reading bulk data (");
  vector<uint8_t> current_data;
  for (int i = 0; i < io_data_width / word_bitwidth; i++) {
    for (int j = 0; j < word_bitwidth / 8; j++) {
      data.push_back(registers[addr][j]);
    }
    current_data = registers[addr];
    out.print("@%d: %s", addr, formatRawDataToWords(current_data).c_str());
    current_data.clear();
    if (i < io_data_width / word_bitwidth - 1) {
      out.print(", ");
    }
    addr++;
  }
  out.print(")\n");

  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteWide);
  dataEvent->size = io_data_width;
  dataEvent->payload = data;

  data_links[0]->send(dataEvent);
}

void RegisterFile::readNarrow() {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  vector<uint8_t> data = registers[port_agus[DataEvent::PortType::ReadNarrow]];
  data.resize(word_bitwidth / 8); // Resize to word size

  dataEvent->size = word_bitwidth;
  dataEvent->payload = data;
  out.output("Reading narrow data (addr=%d, size=%dbits, data=%s)\n",
             port_agus[DataEvent::PortType::ReadNarrow], word_bitwidth,
             formatRawDataToWords(data).c_str());

  data_links[0]->send(dataEvent);
}

void RegisterFile::writeWide() {
  Event *temp_event;
  DataEvent *data_event;
  do {
    temp_event = data_links[0]->recv();
    if (temp_event != nullptr) {
      data_event = dynamic_cast<DataEvent *>(temp_event);
    }
  } while (temp_event != nullptr);
  delete temp_event;

  if (data_event == nullptr)
    out.fatal(CALL_INFO, -1, "Failed to receive data event\n");
  if (data_event->portType != DataEvent::PortType::WriteWide)
    out.fatal(CALL_INFO, -1, "Invalid port type\n");

  // Calculate starting address
  uint32_t addr =
      port_agus[DataEvent::PortType::WriteWide] * io_data_width / word_bitwidth;

  out.output("Writing bulk data (");
  vector<uint8_t> data;
  for (int i = 0; i < data_event->payload.size(); i++) {
    data.push_back(data_event->payload[i]);
    if (data.size() == word_bitwidth / 8) {
      registers[addr] = data;
      out.print("@%d: %s", addr, formatRawDataToWords(data).c_str());
      if (i < data_event->payload.size() - 1) {
        out.print(", ");
      }
      data.clear();
      addr++;
    }
  }
  out.print(")\n");
}

void RegisterFile::writeNarrow() {
  Event *temp_event;
  DataEvent *data_event;
  do {
    temp_event = data_links[0]->recv();
    if (temp_event != nullptr)
      data_event = dynamic_cast<DataEvent *>(temp_event);
  } while (temp_event != nullptr);
  delete temp_event;

  if (data_event == nullptr)
    out.fatal(CALL_INFO, -1, "Failed to receive data event\n");
  if (data_event->portType != DataEvent::PortType::WriteNarrow)
    out.fatal(CALL_INFO, -1, "Invalid port type\n");

  vector<uint8_t> data;
  data.resize(word_bitwidth / 8);
  for (int i = 0; i < word_bitwidth / 8; i++) {
    data[i] = data_event->payload[i];
  }
  registers[port_agus[DataEvent::PortType::WriteNarrow]] = data;

  out.output("Writing narrow data (addr=%d, size=%dbits, data=%s)\n",
             port_agus[DataEvent::PortType::WriteNarrow], word_bitwidth,
             formatRawDataToWords(
                 registers[port_agus[DataEvent::PortType::WriteNarrow]])
                 .c_str());
}