#include <sst/core/component.h>

#include "activationEvent.h"
#include "instructionEvent.h"
#include "memoryEvents.h"
#include "sst/elements/memHierarchy/memEvent.h"
#include "sst/elements/memHierarchy/memEventBase.h"
#include "vec_add.h"

VecAdd::VecAdd(ComponentId_t id, Params &params) : Component(id)
{
    out.init("VecAdd[@p:@l]: ", 16, 0, Output::STDOUT);
    printFrequency = params.find<uint64_t>("printFrequency", 1000);
    clock = params.find<std::string>("clock", "100MHz");
    chunckWidth = params.find<uint8_t>("chunckWidth", 16);
    slot_id = params.find<uint8_t>("slot_id");
    has_io_input_connection = params.find<bool>("has_io_input_connection", false);
    has_io_output_connection = params.find<bool>("has_io_output_connection", false);
    if ((has_io_input_connection || has_io_output_connection) && (slot_id != 1))
    {
        out.fatal(CALL_INFO, -1, "Invalid slot id (only slot 1 is supported for IO connections)\n");
    }

    io_data_width = params.find<uint64_t>("io_data_width", 256);
    for (int i = 0; i < io_data_width / 8; i++)
    {
        dataBuffer.push_back(0); // initialize data buffer
    }

    // Cell coordinates
    std::vector<int> paramsCellCoordinates;
    params.find_array<int>("cell_coordinates", paramsCellCoordinates);
    if (paramsCellCoordinates.size() != 2)
    {
        out.output("Size of cell coordinates: %lu\n", paramsCellCoordinates.size());
        out.fatal(CALL_INFO, -1, "Invalid cell coordinates\n");
    }
    else
    {
        cellCoordinates[0] = paramsCellCoordinates[0];
        cellCoordinates[1] = paramsCellCoordinates[1];
    }

    // Instruction format
    instrBitwidth = params.find<uint64_t>("instr_bitwidth", 32);
    instrTypeBitwidth = params.find<uint64_t>("instr_type_bitwidth", 1);
    instrOpcodeWidth = params.find<uint64_t>("instr_opcode_width", 3);
    instrSlotWidth = params.find<uint64_t>("instr_slot_width", 4);

    // Clock
    TimeConverter *tc = registerClock(clock, new SST::Clock::Handler<VecAdd>(this, &VecAdd::clockTick));

    controllerLink = configureLink("controller_port", "0ns", new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
    out.output("VecAdd: Connected to controller\n");

    // Links
    out.output("%d,%d\n", has_io_input_connection, has_io_output_connection);
    if (has_io_input_connection)
    {
        inputBufferLink = configureLink("input_buffer_port", "0ns", new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
        out.output("VecAdd: Connected to input buffer\n");
    }
    if (has_io_output_connection)
    {
        outputBufferLink = configureLink("output_buffer_port", "0ns", new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
        out.output("VecAdd: Connected to output buffer\n");
    }

    if ((!inputBufferLink) || (!outputBufferLink))
    {
        out.fatal(CALL_INFO, -1, "Invalid IO buffer input and output connections.\nThis component needs IO connections to run.\n");
    }
    else
    {
        out.output("VecAdd: IO connections configured\n");
    }
}

VecAdd::~VecAdd() {}

void VecAdd::init(unsigned int phase)
{
    out.verbose(CALL_INFO, 1, 0, "VecAdd initialized\n");
}

void VecAdd::setup()
{
    // out.verbose(CALL_INFO, 1, 0, "VecAdd setup\n");
}

void VecAdd::complete(unsigned int phase)
{
    // out.verbose(CALL_INFO, 1, 0, "VecAdd completed\n");
}

void VecAdd::finish()
{
    // out.verbose(CALL_INFO, 1, 0, "VecAdd finishing\n");
}

bool VecAdd::clockTick(Cycle_t currentCycle)
{
    if (currentCycle % printFrequency == 0)
    {
        out.output("--- VECADD CYCLE %" PRIu64 " ---\n", currentCycle);
    }
    switch (state)
    {
    case RESET:
        state = IDLE;
        break;
    case IDLE:
        if (active)
        {
            // Check if there is an instruction to decode
            if (instrBuffer == 0)
            {
                out.fatal(CALL_INFO, -1, "No instruction to decode\n");
            }
            // Decode instruction
            instr = decodeInstruction(instrBuffer);
            if (instr.en)
            {
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

void VecAdd::handleEvent(Event *event)
{
    Event *newEvent = event;

    if (newEvent)
    {
        // Check if the event is an ActEvent
        ActEvent *actEvent = dynamic_cast<ActEvent *>(newEvent);
        if (actEvent)
        {
            out.output("VecAdd received ActEvent\n");
            active = true;
        }

        // Check if the event is an InstrEvent
        InstrEvent *instrEvent = dynamic_cast<InstrEvent *>(newEvent);
        if (instrEvent)
        {
            instrBuffer = instrEvent->instruction;
            out.output("VecAdd received InstrEvent: %08x\n", instrBuffer);
        }

        // Check if the event is a memory request
        MemoryEvent *readReq = dynamic_cast<MemoryEvent *>(newEvent);
        if (readReq)
        {
            // Read from memory
            handleMemoryEvent(readReq);
        }

        // Delete the event
        delete newEvent;
    }
}

void VecAdd::handleMemoryEvent(MemoryEvent *memEvent)
{
    ReadResponse *readResp = dynamic_cast<ReadResponse *>(memEvent);
    if (readResp)
    {
        out.output("VecAdd received read response\n");
        out.output("VecAdd dataBuffer (%d bits) = ", io_data_width);
        for (int i = io_data_width / 8 - 1; i >= 0; i--)
        {
            dataBuffer[i] = readResp->data[i];
            out.output("%08b", dataBuffer[i]);
            if (i % 2 == 0)
            {
                out.output(" ");
            }
        }
        out.output("\n");
    }
}

VecAdd::VecAddInstr VecAdd::decodeInstruction(uint32_t instruction)
{
    uint8_t enSegmentLength = 1;
    uint8_t addrSegmentLength = 16;

    // Check opcode
    uint8_t opcode = (instruction & ((1 << instrOpcodeWidth) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth);
    if (opcode != 0)
    {
        out.fatal(CALL_INFO, -1, "Invalid opcode (only 0 is supported)\n");
    }

    VecAddInstr instr;
    instr.en = (instruction & ((1 << enSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - instrSlotWidth - enSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - instrSlotWidth - enSegmentLength);
    instr.addr = (instruction & ((1 << addrSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - instrSlotWidth - enSegmentLength - addrSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - instrSlotWidth - enSegmentLength - addrSegmentLength);

    out.output("VecAdd decoded instruction: en=%d, addr=%d\n", instr.en, instr.addr);

    return instr;
}

void VecAdd::read_from_io()
{
    out.output("VecAdd reading from input buffer\n");
    // Interfaces::StandardMem::Request *req;
    // Interfaces::StandardMem::Addr addr = instr.addr;
    ReadRequest *readReq = new ReadRequest();
    readReq->address = instr.addr;
    readReq->size = io_data_width / 8;
    readReq->column_id = cellCoordinates[1];
    inputBufferLink->send(readReq);
    out.output("VecAdd sent read request\n");
}

void VecAdd::compute_addition()
{
    // Iterate over the data buffer in chunks of chunckWidth
    for (int i = 0; i < io_data_width; i += chunckWidth / 8)
    {
        uint64_t chunk = 0;
        // Combine bytes into a single chunk
        for (int j = 0; j < chunckWidth / 8; ++j)
        {
            chunk |= (static_cast<uint64_t>(dataBuffer[i + j]) << (j * 8));
        }
        // Add 1 to the chunk (ignore overflow)
        chunk++;
        // Mask the chunk to fit within chunckWidth bits
        chunk &= ((1ULL << chunckWidth) - 1);
        // Split the chunk back into bytes and store them in the data buffer
        for (int j = 0; j < chunckWidth / 8; ++j)
        {
            dataBuffer[i + j] = (chunk >> (j * 8)) & 0xFF;
        }
    }
}

void VecAdd::write_to_io()
{
    out.output("VecAdd writing to output buffer\n");
    // Interfaces::StandardMem::Request *req;
    // Interfaces::StandardMem::Addr addr = instr.addr;
    WriteRequest *writeReq = new WriteRequest();
    writeReq->address = instr.addr;
    for (int i = 0; i < io_data_width / 8; i++)
    {
        writeReq->data.push_back(dataBuffer[i]);
    }
    outputBufferLink->send(writeReq);
    out.output("VecAdd sent write request\n");
}