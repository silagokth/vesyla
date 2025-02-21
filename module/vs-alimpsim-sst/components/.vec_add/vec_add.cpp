#include <sst/core/component.h>

#include "activationEvent.h"
#include "instructionEvent.h"
#include "ioEvents.h"
#include "vec_add.h"

VecAdd::VecAdd(ComponentId_t id, Params &params) : DRRAResource(id, params) {
  // Clock
  TimeConverter *tc = registerClock(
      clock, new SST::Clock::Handler<VecAdd>(this, &VecAdd::clockTick));

  chunckWidth = params.find<uint8_t>("chunckWidth", 16);
  if ((has_io_input_connection || has_io_output_connection) && (slot_id != 1)) {
    out.fatal(
        CALL_INFO, -1,
        "Invalid slot id (only slot 1 is supported for IO connections)\n");
  }

  // Links
  controllerLink =
      configureLink("controller_port", "0ns",
                    new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
  out.output("Connected to controller\n");

  if (has_io_input_connection) {
    inputBufferLink =
        configureLink("input_buffer_port", "0ns",
                      new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
    out.output("Connected to input buffer\n");
  }
  if (has_io_output_connection) {
    outputBufferLink =
        configureLink("output_buffer_port", "0ns",
                      new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
    out.output("Connected to output buffer\n");
  }

  if ((!inputBufferLink) || (!outputBufferLink)) {
    out.fatal(CALL_INFO, -1,
              "Invalid IO buffer input and output connections.\nThis component "
              "needs IO connections to run.\n");
  } else {
    out.output("IO connections configured\n");
  }
}

VecAdd::~VecAdd() {}

void VecAdd::init(unsigned int phase) {
  dataBuffer.resize(io_data_width / 8, 0); // Ensure proper size and alignment
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void VecAdd::setup() { out.verbose(CALL_INFO, 1, 0, "Setup\n"); }

void VecAdd::complete(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Completed\n");
}

void VecAdd::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool VecAdd::clockTick(Cycle_t currentCycle) {
  if (currentCycle % printFrequency == 0) {
    out.output("--- VECADD CYCLE %" PRIu64 " ---\n", currentCycle);
  }
  switch (state) {
  case RESET:
    state = IDLE;
    break;
  case IDLE:
    if (isActive) {
      // Check if there is an instruction to decode
      if (instrBuffer == 0) {
        out.fatal(CALL_INFO, -1, "No instruction to decode\n");
      }
      // Decode instruction
      instr = decodeInstruction(instrBuffer);
      if (instr.en) {
        state = COMPUTE_0; // start computation (read from io)
      }
      instrBuffer = 0;
    }
    break;
  case COMPUTE_0: // read from io
    read_from_io();
    state = COMPUTE_1;
    break;
  case COMPUTE_1:
    compute_addition();
    write_to_io();
    instrBuffer = 0;
    state = IDLE;
    return true;
    break;

  default:
    out.fatal(CALL_INFO, -1, "Invalid state\n");
    break;
  }

  return false;
}

void VecAdd::handleEvent(Event *event) {
  if (event) {
    // Check if the event is an ActEvent
    ActEvent *actEvent = dynamic_cast<ActEvent *>(event);
    if (actEvent) {
      out.output("Received ActEvent\n");
      isActive = true;
      return;
    }

    // Check if the event is an InstrEvent
    InstrEvent *instrEvent = dynamic_cast<InstrEvent *>(event);
    if (instrEvent) {
      instrBuffer = instrEvent->instruction;
      out.output("Received InstrEvent: %08x\n", instrBuffer);
      return;
    }

    // Check if the event is a memory request
    IOEvent *readReq = dynamic_cast<IOEvent *>(event);
    if (readReq) {
      // Read from memory
      handleMemoryEvent(readReq);
      return;
    }
  }
}

void VecAdd::handleMemoryEvent(IOEvent *memEvent) {
  IOReadResponse *readResp = dynamic_cast<IOReadResponse *>(memEvent);
  if (readResp) {
    out.output("Received read response\n");
    out.output("DataBuffer (%d bits) = ", io_data_width);
    for (int32_t i = io_data_width / 8 - 1; i >= 0; --i) {
      if (i >= dataBuffer.size()) {
        out.output("Starting index: %d\n", io_data_width / 8 - 1);
        out.output("Size of data buffer: %lu\n", dataBuffer.size());
        out.output("Index: %d\n", i);
        out.fatal(CALL_INFO, -1, "Invalid data buffer size (overflow)\n");
      }
      dataBuffer[i] = readResp->data[i];
      out.print("%08b", dataBuffer[i]);
      if ((i % 2 == 0) && (i != 0)) {
        out.print(" ");
      }
    }
    out.print("\n");
    out.output("Read data from memory\n");
  }
}

VecAdd::VecAddInstr VecAdd::decodeInstruction(uint32_t instruction) {
  uint8_t enSegmentLength = 1;
  uint8_t addrSegmentLength = 16;

  // Check opcode
  uint32_t opcode = getInstrOpcode(instruction);
  if (opcode != 0) {
    out.fatal(CALL_INFO, -1, "Invalid opcode (only 0 is supported)\n");
  }

  VecAddInstr instr;
  instr.en =
      getInstrField(instruction, enSegmentLength,
                    instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                        instrSlotWidth - enSegmentLength);
  instr.addr =
      getInstrField(instruction, addrSegmentLength,
                    instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                        instrSlotWidth - enSegmentLength - addrSegmentLength);

  out.output("Decoded instruction: en=%d, addr=%d\n", instr.en, instr.addr);

  return instr;
}

void VecAdd::read_from_io() {
  out.output("Reading from input buffer\n");

  // Create read request
  IOReadRequest *readReq = new IOReadRequest();
  readReq->address = instr.addr;
  readReq->size = io_data_width / 8;
  readReq->column_id = cell_coordinates[1];

  // Send read request
  inputBufferLink->send(readReq);

  out.output("Sent read request\n");
}

void VecAdd::compute_addition() {
  // Iterate over the data buffer in chunks of chunckWidth
  uint64_t chunckWidthInBytes = chunckWidth / 8;
  uint64_t io_data_widthInBytes = io_data_width / 8;
  for (int i = 0; i < io_data_widthInBytes - chunckWidth;
       i += chunckWidthInBytes) {
    uint64_t chunk = 0;
    // Combine bytes into a single chunk
    for (int j = 0; j < chunckWidthInBytes; j++) {
      if (i + j >= dataBuffer.size()) {
        out.output("1Starting index: %d\n", i);
        out.output("Stopping condition: j < %d\n", chunckWidthInBytes);
        out.output("Size of data buffer: %lu\n", dataBuffer.size());
        out.output("Index: %d\n", i + j);
        out.fatal(CALL_INFO, -1, "Invalid data buffer size (overflow)\n");
      }
      chunk |= (dataBuffer[i + j]) << (j * 8);
    }
    // Add 1 to the chunk (ignore overflow)
    chunk++;
    // Mask the chunk to fit within chunckWidth bits
    chunk &= ((1ULL << chunckWidth) - 1);
    // Split the chunk back into bytes and store them in the data buffer
    for (int j = 0; j < chunckWidthInBytes; j++) {
      if (i + j >= dataBuffer.size()) {
        out.output("2Starting index: %d\n", i);
        out.output("Stopping condition: j < %d\n", chunckWidthInBytes);
        out.output("Size of data buffer: %lu\n", dataBuffer.size());
        out.output("Index: %d\n", i + j);
        out.fatal(CALL_INFO, -1, "Invalid data buffer size (overflow)\n");
      }
      dataBuffer[i + j] = (chunk >> (j * 8)) & 0xFF;
    }
  }
}

void VecAdd::write_to_io() {
  out.output("Writing to output buffer\n");
  // Interfaces::StandardMem::Request *req;
  // Interfaces::StandardMem::Addr addr = instr.addr;
  IOWriteRequest *writeReq = new IOWriteRequest();
  writeReq->address = instr.addr;
  for (int i = 0; i < io_data_width / 8; i++) {
    writeReq->data.push_back(dataBuffer[i]);
  }
  outputBufferLink->send(writeReq);
  out.output("Sent write request\n");
}