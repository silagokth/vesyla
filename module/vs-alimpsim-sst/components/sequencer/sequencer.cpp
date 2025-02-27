#include <fstream>
#include <sst/core/component.h>
#include <sst/core/link.h>

#include "activationEvent.h"
#include "instructionEvent.h"
#include "sequencer.h"
#include <bitset>

Sequencer::Sequencer(ComponentId_t id, Params &params)
    : DRRAController(id, params) {
  assemblyProgramPath = params.find<std::string>("assembly_program_path");

  // Register as primary component
  registerAsPrimaryComponent();
  primaryComponentDoNotEndSim();
}

Sequencer::~Sequencer() {}

void Sequencer::init(unsigned int phase) {
  // Load the assembly program
  load_assembly_program(assemblyProgramPath);

  // Initialize scalar and bool registers
  for (uint32_t i = 0; i < num_slots; i++) {
    scalarRegisters.push_back(0);
    boolRegisters.push_back(false);
  }

  // End of initialization
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void Sequencer::setup() {}

void Sequencer::complete(unsigned int phase) {}

void Sequencer::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool Sequencer::clockTick(Cycle_t currentCycle) {
  if (currentCycle % 10 == 0) {
    if (cyclesToWait > 0) {
      cyclesToWait--;
      out.output("Waiting %u cycles\n", cyclesToWait);
      return false;
    }

    if (pc >= assemblyProgram.size()) {
      out.fatal(CALL_INFO, -1, "Program counter out of bounds\n");
    }
    uint32_t instruction = assemblyProgram[pc];
    fetch_decode(instruction);

    if (readyToFinish) {
      primaryComponentOKToEndSim();
      return true;
    }
  }
  return false;
}

void Sequencer::load_assembly_program(std::string assemblyProgramPath) {
  // Load the assembly program
  if (assemblyProgramPath.empty()) {
    out.fatal(CALL_INFO, -1, "No assembly program provided\n");
  }
  std::ifstream assemblyProgramFile(assemblyProgramPath);
  if (!assemblyProgramFile.is_open()) {
    out.fatal(CALL_INFO, -1, "Failed to open assembly program file\n");
  }

  std::string line;
  bool isSelfCell = false;
  while (std::getline(assemblyProgramFile, line)) {
    // out.output("Read line: %s\n", line.c_str());
    if (line.find("cell") != std::string::npos) {
      out.output("Found cell section in line: %s\n", line.c_str());
      if (line.find("cell " + std::to_string(cell_coordinates[1]) + " " +
                    std::to_string(cell_coordinates[0])) != std::string::npos) {
        isSelfCell = true;
      } else {
        isSelfCell = false;
      }
      continue;
    } else if (isSelfCell) {
      out.output("Adding instruction: %s\n", line.c_str());
      std::bitset<32> bits(line);
      assemblyProgram.push_back(static_cast<uint32_t>(bits.to_ulong()));
    }
  }
  assemblyProgram.shrink_to_fit(); // Ensure proper alignment and size
  out.output("Loaded %lu instructions\n", assemblyProgram.size());
}

void Sequencer::fetch_decode(uint32_t instruction) {
  // Decode instruction
  // TODO make dependent on ISA.json
  if (instrBitwidth != 32) {
    out.fatal(CALL_INFO, -1,
              "Invalid instruction bitwidth. Only 32-bit "
              "is supported for now.\n");
  }
  uint32_t instruction_type = getInstrType(instruction);
  uint32_t opcode = getInstrOpcode(instruction);
  uint32_t slot = getInstrSlot(instruction);

  if (instruction_type == 1) // Send event to resource
  {
    InstrEvent *event = new InstrEvent();
    event->instruction = instruction;
    out.output("Sending INSTRUCTION event to slot %u\n", slot);
    if (!slot_links[slot]->isConfigured()) {
      out.fatal(CALL_INFO, -1, "Slot link %u not configured\n", slot);
    }
    slot_links[slot]->send(event);
  } else {
    switch (opcode) {
    case 0: // HALT
      halt();
      break;

    case 1: // WAIT
      wait(instruction);
      break;

    case 2: // ACT
      activate(instruction);
      break;

    case 3: // CALC
      out.fatal(CALL_INFO, -1, "TODO implement CALC\n");
      calculate(instruction);
      break;

    case 4: // BRN
      out.fatal(CALL_INFO, -1, "TODO implement BRN\n");
      branch(instruction);
      break;

    default:
      break;
    }
  }

  // Increment program counter
  pc++;
}

void Sequencer::halt() {
  out.output("HALT\n");
  readyToFinish = true;
}

void Sequencer::wait(uint32_t instr) {
  out.output("WAIT\n");
  uint32_t mode_segment_length = 1;
  uint32_t cycleSegmentLength = 27;

  // Check validity
  if (mode_segment_length + cycleSegmentLength >
      instrBitwidth - instrTypeBitwidth - instrOpcodeWidth) {
    out.fatal(CALL_INFO, -1, "Invalid instruction format\n");
  }

  // Extract segments
  uint32_t mode = (instr & ((1 << mode_segment_length) - 1)
                               << (instrBitwidth - instrTypeBitwidth -
                                   instrOpcodeWidth - mode_segment_length)) >>
                  (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                   mode_segment_length);
  uint32_t cycles = (instr & ((1 << cycleSegmentLength) - 1));

  if (mode == 0) {
    wait_cycles(cycles);
  } else {
    wait_event();
  }

  out.output("WAIT: mode=%s, cycles=%u\n",
             std::bitset<1>(mode).to_string().c_str(), cycles);
}

void Sequencer::wait_cycles(uint32_t cycles) { cyclesToWait = cycles; }

void Sequencer::wait_event() { out.output("WAIT EVENT\n"); }

void Sequencer::activate(uint32_t instr) {
  uint32_t ports_segment_length = 16;
  uint32_t mode_segment_length = 4;
  uint32_t param_segment_length = 8;

  // Extract segments
  uint32_t ports = getInstrField(instr, ports_segment_length, 12);
  uint32_t mode = getInstrField(instr, mode_segment_length, 8);
  uint32_t param = getInstrField(instr, param_segment_length, 0);

  uint32_t target_ports_for_slot = 0;
  uint32_t temp_ports = ports;
  uint32_t current_slot = 0;

  switch (mode) {
  case 0: // Continuous ports starting from slot X (param)
    current_slot = param;
    for (uint32_t i = 0; i < ports_segment_length; i += 4) {
      // Extract 1-bit activation for each port and accumulate
      target_ports_for_slot = temp_ports & 0b1111;
      temp_ports >>= 4;

      // Skip this slot if no ports are activated or not linked
      if ((target_ports_for_slot == 0) ||
          (slot_links[current_slot] == nullptr)) {
        current_slot++;
        continue;
      }

      // Send on-hot encoded 4-bits activation to the slot
      ActEvent *event = new ActEvent();
      event->slot_id = current_slot;
      event->ports = target_ports_for_slot;
      slot_links[current_slot]->send(event);

      // Print debug info
      out.output("act instr: %s (ports=%d, mode=%d, param=%d)\n",
                 std::bitset<32>(instr).to_string().c_str(), ports, mode,
                 param);
      out.output("act (slot=%u, mode=%d, param=%d, ports=%s)\n", current_slot,
                 mode, param,
                 std::bitset<4>(target_ports_for_slot).to_string().c_str());

      // Next slot
      target_ports_for_slot = 0;
      current_slot++;
    }
    break;

  default:
    out.fatal(CALL_INFO, -1, "ACT mode not implemented\n");
    break;
  }
}

void Sequencer::calculate(uint32_t instr) {
  uint32_t mode_segment_length = 6;
  uint32_t operand1SegmentLength = 4;
  uint32_t operand2SDSegmentLength = 1;
  uint32_t operand2SegmentLength = 8;
  uint32_t resultSegmentLength = 4;

  // Check validity
  if (mode_segment_length + operand1SegmentLength + operand2SDSegmentLength +
          operand2SegmentLength + resultSegmentLength >
      instrBitwidth - instrTypeBitwidth - instrOpcodeWidth) {
    out.fatal(CALL_INFO, -1, "Invalid instruction format\n");
  }

  // Extract segments
  uint32_t mode = getInstrField(instr, mode_segment_length,
                                instrBitwidth - instrTypeBitwidth -
                                    instrOpcodeWidth - mode_segment_length);
  uint32_t operand1 =
      getInstrField(instr, operand1SegmentLength,
                    instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                        mode_segment_length - operand1SegmentLength);
  uint32_t operand2SD =
      getInstrField(instr, operand2SDSegmentLength,
                    instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                        mode_segment_length - operand1SegmentLength -
                        operand2SDSegmentLength);
  uint32_t operand2 =
      getInstrField(instr, operand2SegmentLength,
                    instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                        mode_segment_length - operand1SegmentLength -
                        operand2SDSegmentLength - operand2SegmentLength);
  uint32_t result =
      getInstrField(instr, resultSegmentLength,
                    instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                        mode_segment_length - operand1SegmentLength -
                        operand2SDSegmentLength - operand2SegmentLength -
                        resultSegmentLength);

  std::string operationStr;

  switch (mode) {
  case 0:
    operationStr = "idle";
    break;

  case 1:
    operationStr = "add";
    scalarRegisters[result] = operand1 + operand2;
    break;
  case 2:
    operationStr = "sub";
    scalarRegisters[result] = operand1 - operand2;
    break;
  case 3:
    operationStr = "lls";
    scalarRegisters[result] = operand1 / (1 << operand2);
    break;
  case 4:
    operationStr = "lrs";
    scalarRegisters[result] = operand1 * (1 << operand2);
    break;
  case 5:
    operationStr = "mul";
    scalarRegisters[result] = operand1 * operand2;
    break;
  case 6:
    operationStr = "div";
    scalarRegisters[result] = operand1 / operand2;
    break;
  case 7:
    operationStr = "mod";
    scalarRegisters[result] = operand1 % operand2;
    break;
  case 8:
    operationStr = "bitand";
    scalarRegisters[result] = operand1 & operand2;
    break;
  case 9:
    operationStr = "bitor";
    scalarRegisters[result] = operand1 | operand2;
    break;
  case 10:
    operationStr = "bitinv";
    scalarRegisters[result] = ~operand1;
    break;
  case 11:
    operationStr = "bitxor";
    scalarRegisters[result] = operand1 ^ operand2;
    break;
  case 17:
    operationStr = "eq";
    boolRegisters[result] = operand1 == operand2;
    break;
  case 18:
    operationStr = "ne";
    boolRegisters[result] = operand1 != operand2;
    break;
  case 19:
    operationStr = "gt";
    boolRegisters[result] = operand1 > operand2;
    break;
  case 20:
    operationStr = "ge";
    boolRegisters[result] = operand1 >= operand2;
    break;
  case 21:
    operationStr = "lt";
    boolRegisters[result] = operand1 < operand2;
    break;
  case 22:
    operationStr = "le";
    boolRegisters[result] = operand1 <= operand2;
    break;
  case 32:
    operationStr = "and";
    boolRegisters[result] =
        boolRegisters[operand1] && boolRegisters[operand2]; // TODO check this
    break;
  case 33:
    operationStr = "or";
    boolRegisters[result] = boolRegisters[operand1] || boolRegisters[operand2];
    break;
  case 34:
    operationStr = "not";
    boolRegisters[result] = !boolRegisters[operand1];
    break;

  default:
    out.fatal(CALL_INFO, -1, "Invalid operation mode\n");
    break;
  }

  out.output("CALCULATE: mode=%s (%s), operand1=%s, "
             "operand2SD=%s, "
             "operand2=%s, result=%s\n",
             std::bitset<6>(mode).to_string().c_str(), operationStr.c_str(),
             std::bitset<4>(operand1).to_string().c_str(),
             std::bitset<1>(operand2SD).to_string().c_str(),
             std::bitset<8>(operand2).to_string().c_str(),
             std::bitset<4>(result).to_string().c_str());
}

void Sequencer::branch(uint32_t instr) {
  uint32_t regSegmentLength = 4;
  int32_t targetTrueSegmentLength = 9;
  int32_t targetFalseSegmentLength = 9;

  // Check validity
  if (regSegmentLength + targetTrueSegmentLength + targetFalseSegmentLength >
      instrBitwidth - instrTypeBitwidth - instrOpcodeWidth) {
    out.fatal(CALL_INFO, -1, "Invalid instruction format\n");
  }

  // Extract segments
  uint32_t reg = getInstrField(instr, regSegmentLength,
                               instrBitwidth - instrTypeBitwidth -
                                   instrOpcodeWidth - regSegmentLength);
  int32_t targetTrue =
      getInstrField(instr, targetTrueSegmentLength,
                    instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                        regSegmentLength - targetTrueSegmentLength);
  int32_t targetFalse = getInstrField(
      instr, targetFalseSegmentLength,
      instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - regSegmentLength -
          targetTrueSegmentLength - targetFalseSegmentLength);

  // Sign extend targetTrue and targetFalse
  if (targetTrue & (1 << (targetTrueSegmentLength - 1))) {
    targetTrue |= ~((1 << targetTrueSegmentLength) - 1);
  }
  if (targetFalse & (1 << (targetFalseSegmentLength - 1))) {
    targetFalse |= ~((1 << targetFalseSegmentLength) - 1);
  }

  // Compute new PC
  if (boolRegisters[reg]) {
    pc += targetTrue;
  } else {
    pc += targetFalse;
  }

  out.output("BRANCH: reg=%s, targetTrue=%d, targetFalse=%d\n",
             std::bitset<4>(reg).to_string().c_str(), targetTrue, targetFalse);
}