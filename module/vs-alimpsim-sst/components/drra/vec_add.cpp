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

    io_data_width = params.find<uint64_t>("io_data_width", 256);
    for (int i = 0; i < io_data_width / 8; i++)
    {
        dataBuffer.push_back(0); // initialize data buffer
    }

    // Instruction format
    instrBitwidth = params.find<uint64_t>("instr_bitwidth", 32);
    instrTypeBitwidth = params.find<uint64_t>("instr_type_bitwidth", 1);
    instrOpcodeWidth = params.find<uint64_t>("instr_opcode_width", 3);
    instrSlotWidth = params.find<uint64_t>("instr_slot_width", 4);

    // Clock
    TimeConverter *tc = registerClock(clock, new SST::Clock::Handler<VecAdd>(this, &VecAdd::clockTick));

    // Links
    inputBufferLink = configureLink("input_buffer_port", "0ns", new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
    out.output("VecAdd: Connected to input buffer\n");
    outputBufferLink = configureLink("output_buffer_port", "0ns", new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
    out.output("VecAdd: Connected to output buffer\n");

    // Interface to input and output buffers
    // inputBuffer = loadUserSubComponent<Interfaces::StandardMem>("input_buffer", ComponentInfo::SHARE_NONE, tc, new Interfaces::StandardMem::Handler<VecAdd>(this, &VecAdd::handleMemoryEvent));
    // if (!inputBuffer)
    // {
    //     SST::Params interfaceParams;
    //     interfaceParams.insert("port", "input_buffer_port");
    //     inputBuffer = loadAnonymousSubComponent<Interfaces::StandardMem>("memHierarchy.standardInterface", "memory", 0, ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, interfaceParams, tc, new Interfaces::StandardMem::Handler<VecAdd>(this, &VecAdd::handleMemoryEvent));
    // }
    // sst_assert(inputBuffer, CALL_INFO, -1, "Failed to load input buffer\n");

    // // outputBuffer = loadUserSubComponent<Interfaces::StandardMem>("output_buffer", ComponentInfo::SHARE_NONE, tc, new Interfaces::StandardMem::Handler<VecAdd>(this, &VecAdd::handleMemoryEvent));
    // if (!outputBuffer)
    // {
    //     SST::Params interfaceParams;
    //     interfaceParams.insert("port", "output_buffer_port");
    //     outputBuffer = loadAnonymousSubComponent<Interfaces::StandardMem>("memHierarchy.standardInterface", "memory", 0, ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, interfaceParams, tc, new Interfaces::StandardMem::Handler<VecAdd>(this, &VecAdd::handleMemoryEvent));
    // }
    // sst_assert(outputBuffer, CALL_INFO, -1, "Failed to load output buffer\n");
    // out.output("VecAdd (ID:%lu): Connected to input and output buffers\n", id);
}

VecAdd::~VecAdd() {}

void VecAdd::init(unsigned int phase)
{
    controllerLink = configureLink("controller_port", "0ns", new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
    out.output("VecAdd: Connected to controller\n");
    // inputBuffer->init(phase);
    // outputBuffer->init(phase);
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
        break;

    default:
        out.fatal(CALL_INFO, -1, "Invalid state\n");
        break;
    }

    return true;
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

        // Delete the event
        delete newEvent;
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

void VecAdd::handleMemoryEvent(Interfaces::StandardMem::Request *response)
{
    Interfaces::StandardMem::ReadResp *readResp = dynamic_cast<Interfaces::StandardMem::ReadResp *>(response);
    if (readResp)
    {
        out.output("VecAdd received read response\n");
        for (int i = 0; i < io_data_width / 8; i++)
        {
            dataBuffer[i] = readResp->data[i];
        }
    }
    delete response;
}

void VecAdd::read_from_io()
{
    out.output("VecAdd reading from input buffer\n");
    // Interfaces::StandardMem::Request *req;
    // Interfaces::StandardMem::Addr addr = instr.addr;
    ReadRequest *readReq = new ReadRequest();
    readReq->address = instr.addr;
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
        for (int j = 0; j < chunckWidth; ++j)
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