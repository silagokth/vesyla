// filepath:
// /home/paul/Develop/vesyla-suite-4/module/vs-alimpsim-sst/components/dpu/dpu.cpp
#include "dpu.h"

#include "activationEvent.h"
#include "dataEvent.h"
#include "instructionEvent.h"

#include <cmath>

using namespace SST;

DPU::DPU(SST::ComponentId_t id, SST::Params &params)
    : DRRAResource(id, params) {
  // Clock
  SST::TimeConverter *tc =
      registerClock(clock, new SST::Clock::Handler<DPU>(this, &DPU::clockTick));
}

void DPU::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void DPU::setup() { out.verbose(CALL_INFO, 1, 0, "Setup\n"); }

void DPU::complete(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Completed\n");
}

void DPU::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool DPU::clockTick(SST::Cycle_t currentCycle) {
  if (currentCycle % 10 == 0) {
    out.output("--- DPU CYCLE %" PRIu64 " ---\n", currentCycle / 10);
  }

  executeScheduledEventsForCycle(currentCycle);

  // Execute DPU operation
  // fsmHandlers[current_fsm]();

  // // Increment the active cycle
  // activeCycle++;
  return false;
}

void DPU::handleEvent(SST::Event *event) {
  if (event) {
    // Check if the event is an ActEvent
    ActEvent *actEvent = dynamic_cast<ActEvent *>(event);
    if (actEvent) {
      out.output("Received ActEvent\n");
      activatePortsForSlot(actEvent->slot_id, actEvent->ports);
      return;
    }

    // Check if the event is an InstrEvent
    InstrEvent *instrEvent = dynamic_cast<InstrEvent *>(event);
    if (instrEvent) {
      uint32_t instr = instrEvent->instruction;
      decodeInstr(instr);
      out.output("Received InstrEvent: %08x\n", instr);
      return;
    }
  }
}

void DPU::handleEventWithSlotID(SST::Event *event, uint32_t slot_id) {
  DataEvent *dataEvent = dynamic_cast<DataEvent *>(event);
  if (dataEvent) {
    bool anyPortActive = false;
    for (const auto &port : active_ports) {
      if (isPortActive(port.first)) {
        anyPortActive = true;
        break;
      }
    }
    if (!anyPortActive) {
      out.fatal(CALL_INFO, -1, "Received data while inactive\n", slot_id);
    }

    // get the data from the event
    uint64_t data = 0;
    for (int i = 0; i < dataEvent->size; i++) {
      data |= dataEvent->payload[i] << (i * 8);
    }

    // store the data in the corresponding buffer
    data_buffers[slot_id] = data;
  }
}

void DPU::decodeInstr(uint32_t instr) {
  uint32_t instrOpcode = getInstrOpcode(instr);

  switch (instrOpcode) {
  case REP:
    handleRep(instr);
    break;
  case REPX:
    handleRepx(instr);
    break;
  case FSM:
    handleFSM(instr);
    break;
  case DPU_OP:
    handleDPU(instr);
    break;
  default:
    out.fatal(CALL_INFO, -1, "Invalid opcode: %u\n", instrOpcode);
  }
}

void DPU::handleRep(uint32_t instr) {
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
    next_timing_states[0].addRepetition(iter, delay, level, step);
  } catch (const std::exception &e) {
    out.fatal(CALL_INFO, -1, "Failed to add repetition: %s\n", e.what());
  }
}

void DPU::handleRepx(uint32_t instr) { handleRep(instr); }

void DPU::handleFSM(uint32_t instr) {
  // Instruction fields
  uint32_t port = getInstrField(instr, 3, 21);
  uint32_t delay_0 = getInstrField(instr, 7, 14);
  uint32_t delay_1 = getInstrField(instr, 7, 7);
  uint32_t delay_2 = getInstrField(instr, 7, 0);
  // TODO: what are the use cases for delay_1 and delay_2?

  // add transition to the timing model
  try {
    next_timing_states[0].addTransition(
        delay_0, "event_" + std::to_string(current_event_number), [this, port] {
          out.output(" FSM switched to %d\n", port);
          current_fsm = port;
        });
    current_event_number++;
  } catch (const std::exception &e) {
    out.fatal(CALL_INFO, -1, "Failed to add transition: %s\n", e.what());
  }
}

void DPU::handleDPU(uint32_t instr) {
  uint32_t option = getInstrField(instr, 2, 22);
  DPU_MODE mode = (DPU_MODE)getInstrField(instr, 5, 17);
  uint32_t immediate = getInstrField(instr, 16, 1);

  // replace the data buffer with the immediate value if needed
  if (mode == DPU_MODE::ADD_CONST || mode == DPU_MODE::SUBT_ABS ||
      mode == DPU_MODE::MULT_CONST || mode == DPU_MODE::MAX_MIN_CONST ||
      mode == DPU_MODE::LD_IR) {
    data_buffers[1] = immediate;
  }

  // Add the event handler
  fsmHandlers[current_fsm] = dsuHandlers[mode];
}

void DPU::handleIdle(void) { out.output("IDLE\n"); }

void DPU::handleAdd(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] + data_buffers[1];
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleSumAcc(void) {
  out.fatal(CALL_INFO, -1, "SUM_ACC not implemented\n");
}

void DPU::handleSubt(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] - data_buffers[1];
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleMode6(void) {
  out.fatal(CALL_INFO, -1, "MODE_6 not implemented\n");
}

void DPU::handleMult(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] * data_buffers[1];
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleMAC(void) { out.fatal(CALL_INFO, -1, "MAC not implemented\n"); }

void DPU::handleAccumulate(void) {
  out.fatal(CALL_INFO, -1, "ACCUMULATE not implemented\n");
}

void DPU::handleLD_IR(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((data_buffers[1] >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleAXPY(void) {
  out.fatal(CALL_INFO, -1, "AXPY not implemented\n");
}

void DPU::handleMaxMinAcc(void) {
  out.fatal(CALL_INFO, -1, "MAX_MIN_ACC not implemented\n");
}

void DPU::handleMode15(void) {
  out.fatal(CALL_INFO, -1, "MODE_15 not implemented\n");
}

void DPU::handleMaxMin(void) {
  out.fatal(CALL_INFO, -1, "MAX_MIN not implemented\n");
}

void DPU::handleShiftL(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] << data_buffers[1];
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleShiftR(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] >> data_buffers[1];
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleSigm(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = 1 / (1 + exp(-data_buffers[0]));
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleTanhyp(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = tanh(data_buffers[0]);
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleExpon(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = exp(data_buffers[0]);
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleLkRelu(void) {
  out.fatal(CALL_INFO, -1, "LK_RELU not implemented\n");
}

void DPU::handleRelu(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] > 0 ? data_buffers[0] : 0;
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleDiv(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] / data_buffers[1];
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleAccSoftmax(void) {
  out.fatal(CALL_INFO, -1, "ACC_SOFTMAX not implemented\n");
}

void DPU::handleDivSoftmax(void) {
  DataEvent *dataEvent = new DataEvent(DataEvent::PortType::WriteNarrow);
  uint64_t result = data_buffers[0] / data_buffers[1];
  result = exp(result) / (exp(data_buffers[0]) + exp(data_buffers[1]));
  dataEvent->size = word_bitwidth;
  for (int i = 0; i < word_bitwidth / 8; i++) {
    dataEvent->payload.push_back((result >> (i * 8)) & 0xFF);
  }
  data_links[0]->send(dataEvent);
}

void DPU::handleLdAcc(void) {
  out.fatal(CALL_INFO, -1, "LD_ACC not implemented\n");
}

void DPU::handleScaleDw(void) {
  out.fatal(CALL_INFO, -1, "SCALE_DW not implemented\n");
}

void DPU::handleScaleUp(void) {
  out.fatal(CALL_INFO, -1, "SCALE_UP not implemented\n");
}

void DPU::handleMacInter(void) {
  out.fatal(CALL_INFO, -1, "MAC_INTER not implemented\n");
}

void DPU::handleMode31(void) {
  out.fatal(CALL_INFO, -1, "MODE_31 not implemented\n");
}