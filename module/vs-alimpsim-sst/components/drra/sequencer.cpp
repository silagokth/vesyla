#include <fstream>
#include <sst/core/component.h>
#include <sst/core/link.h>

#include "activationEvent.h"
#include "instructionEvent.h"
#include "sequencer.h"
#include <bitset>

Sequencer::Sequencer(ComponentId_t id, Params &params) : Component(id)
{
    out.init("Sequencer[@p:@l]: ", 16, 0, Output::STDOUT);
    printFrequency = params.find<uint64_t>("print_frequency", 1);
    numSlots = params.find<uint64_t>("num_slots", 16);
    assemblyProgramPath = params.find<std::string>("assembly_program_path");
    clock = params.find<std::string>("clock", "100MHz");

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

    // Slots ports
    std::string linkPrefix = "slot_port";
    std::string linkName = linkPrefix + "0";
    int portNum = 0;
    while (isPortConnected(linkName))
    {
        SST::Link *link = configureLink(linkName, "1ns", new InstrEvent::Handler<Sequencer>(this, &Sequencer::handleEvent));
        sst_assert(link, CALL_INFO, -1, "Failed to configure link %s\n", linkName.c_str());

        if (!link)
        {
            out.fatal(CALL_INFO, -1, "Failed to configure link %s\n", linkName.c_str());
        }
        slotLinks.push_back(link);

        // Next link
        portNum++;
        linkName = linkPrefix + std::to_string(portNum);
    }

    // Check number of slots
    if (slotLinks.size() > numSlots)
    {
        out.output("Number of slots: %lu\n", slotLinks.size());
        out.fatal(CALL_INFO, -1, "Invalid number of slots\n");
    }
    out.output("Sequencer: Connected %lu slot links\n", slotLinks.size());

    registerClock(clock, new SST::Clock::Handler<Sequencer>(this, &Sequencer::clockTick));

    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

Sequencer::~Sequencer() {}

void Sequencer::init(unsigned int phase)
{
    // Load the assembly program
    load_assembly_program(assemblyProgramPath);

    // Initialize scalar and bool registers
    for (uint32_t i = 0; i < numSlots; i++)
    {
        scalarRegisters.push_back(0);
        boolRegisters.push_back(false);
    }

    // End of initialization
    out.verbose(CALL_INFO, 1, 0, "Sequencer initialized\n");
}

void Sequencer::setup()
{
    out.verbose(CALL_INFO, 1, 0, "Sequencer setup\n");
}

void Sequencer::complete(unsigned int phase)
{
    out.verbose(CALL_INFO, 1, 0, "Sequencer completed\n");
}

void Sequencer::finish()
{
    out.verbose(CALL_INFO, 1, 0, "Sequencer finishing\n");
}

bool Sequencer::clockTick(Cycle_t currentCycle)
{
    if (currentCycle % printFrequency == 0)
    {
        if (cyclesToWait == 0)
        {
            out.output("--- SEQUENCER CYCLE %" PRIu64 " ---\n", currentCycle);
        }
    }

    if (cyclesToWait > 0)
    {
        cyclesToWait--;
        return false;
    }

    uint32_t instruction = assemblyProgram[pc];
    fetch_decode(instruction);

    if (readyToFinish)
    {
        primaryComponentOKToEndSim();
        return true;
    }

    if (pc >= assemblyProgram.size())
    {
        out.fatal(CALL_INFO, -1, "Program counter out of bounds\n");
    }
    return false;
}

void Sequencer::handleEvent(Event *event)
{
    out.output("Sequencer received event\n");
}

void Sequencer::load_assembly_program(std::string assemblyProgramPath)
{
    // Load the assembly program
    if (assemblyProgramPath.empty())
    {
        out.fatal(CALL_INFO, -1, "No assembly program provided\n");
    }
    std::ifstream assemblyProgramFile(assemblyProgramPath);
    if (!assemblyProgramFile.is_open())
    {
        out.fatal(CALL_INFO, -1, "Failed to open assembly program file\n");
    }
    out.output("Loading assembly program from %s\n", assemblyProgramPath.c_str());
    std::string line;
    bool isSelfCell = false;
    while (std::getline(assemblyProgramFile, line))
    {
        out.output("Read line: %s\n", line.c_str());
        if (line.find("cell") != std::string::npos)
        {
            out.output("Found cell section in line: %s\n", line.c_str());
            if (line.find("cell " + std::to_string(cellCoordinates[0]) + "_" + std::to_string(cellCoordinates[1])) != std::string::npos)
            {
                isSelfCell = true;
            }
            else
            {
                isSelfCell = false;
            }
            continue;
        }
        else if (isSelfCell)
        {
            out.output("Adding instruction: %s\n", line.c_str());
            std::bitset<32> bits(line);
            assemblyProgram.push_back(static_cast<uint32_t>(bits.to_ulong()));
        }
    }
    out.output("Loaded %lu instructions\n", assemblyProgram.size());
}

void Sequencer::fetch_decode(uint32_t instruction)
{
    // Decode instruction
    // TODO make dependent on ISA.json
    if (instrBitwidth != 32)
    {
        out.fatal(CALL_INFO, -1, "Invalid instruction bitwidth. Only 32-bit is supported for now.\n");
    }
    uint32_t instruction_type = (instruction & ((1 << instrTypeBitwidth) - 1) << (instrBitwidth - instrTypeBitwidth)) >> (instrBitwidth - instrTypeBitwidth);
    uint32_t opcode = (instruction & ((1 << instrOpcodeWidth) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth);
    uint32_t slot = (instruction & ((1 << instrSlotWidth) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - instrSlotWidth)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - instrSlotWidth);
    uint32_t content =
        (instruction & ((1 << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - instrSlotWidth)) - 1));
    InstrEvent *event = new InstrEvent();

    if (instruction_type == 1) // Send event to resource
    {
        event->instruction = instruction;
        slotLinks[slot]->send(event);
        out.output("Sequencer sent INSTRUCTION event to slot %u\n", slot);
    }
    else
    {
        switch (opcode)
        {
        case 0: // HALT
            halt();
            break;

        case 1: // WAIT
            wait(content);
            break;

        case 2: // ACT
            activate(content);
            break;

        case 3: // CALC
            out.fatal(CALL_INFO, -1, "TODO implement CALC\n");
            calculate(content);
            break;

        case 4: // BRN
            out.fatal(CALL_INFO, -1, "TODO implement BRN\n");
            branch(content);
            break;

        default:
            break;
        }
    }

    // Increment program counter
    pc++;
}

void Sequencer::halt()
{
    out.output("HALT\n");
    readyToFinish = true;
}

void Sequencer::wait(uint32_t content)
{
    uint32_t modeSegmentLength = 1;
    uint32_t cycleSegmentLength = 27;

    // Check validity
    if (modeSegmentLength + cycleSegmentLength > instrBitwidth - instrTypeBitwidth - instrOpcodeWidth)
    {
        out.fatal(CALL_INFO, -1, "Invalid instruction format\n");
    }

    // Extract segments
    uint32_t mode = (content & ((1 << modeSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength);
    uint32_t cycles = (content & ((1 << cycleSegmentLength) - 1));

    if (mode == 0)
    {
        wait_cycles(cycles);
    }
    else
    {
        wait_event();
    }

    out.output("WAIT: mode=%s, cycles=%u\n", std::bitset<1>(mode).to_string().c_str(), cycles);
}

void Sequencer::wait_cycles(uint32_t cycles)
{
    cyclesToWait = cycles;
}

void Sequencer::wait_event()
{
    out.output("WAIT EVENT\n");
}

void Sequencer::activate(uint32_t content)
{
    uint32_t slotsSegmentLength = 16;
    uint32_t modeSegmentLength = 4;
    uint32_t paramSegmentLength = 8;

    // Check validity
    if (slotsSegmentLength + modeSegmentLength + paramSegmentLength > instrBitwidth - instrTypeBitwidth - instrOpcodeWidth)
    {
        out.fatal(CALL_INFO, -1, "Invalid instruction format\n");
    }

    // Extract segments
    uint32_t slots = (content & ((1 << slotsSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - slotsSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - slotsSegmentLength);
    uint32_t mode = (content & ((1 << modeSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - slotsSegmentLength - modeSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - slotsSegmentLength - modeSegmentLength);
    uint32_t param = (content & ((1 << paramSegmentLength) - 1));

    uint32_t targetPort = 0;

    // Send activate event to ports (one-hot encoded)
    for (uint32_t i = 0; i < numSlots; i++)
    {
        if (slots & (1 << i))
        {
            switch (mode)
            {
            case 0:
                targetPort = param + i;
                break;
            case 1:
                targetPort = param;
                break;
            case 2:
                out.fatal(CALL_INFO, -1, "ACT MODE 2 not implemented\n");
                break;

            default:
                out.fatal(CALL_INFO, -1, "Invalid mode\n");
                break;
            }
            ActEvent *event = new ActEvent();
            event->port = targetPort;
            slotLinks[i]->send(event);
            out.output("Sequencer sent ACTIVATE event to slot %u, port %u\n", i, targetPort);
        }
    }

    out.output("ACTIVATE: slots=%s, mode=%s, param=%s\n", std::bitset<16>(slots).to_string().c_str(), std::bitset<4>(mode).to_string().c_str(), std::bitset<8>(param).to_string().c_str());
}

void Sequencer::calculate(uint32_t content)
{
    uint32_t modeSegmentLength = 6;
    uint32_t operand1SegmentLength = 4;
    uint32_t operand2SDSegmentLength = 1;
    uint32_t operand2SegmentLength = 8;
    uint32_t resultSegmentLength = 4;

    // Check validity
    if (modeSegmentLength + operand1SegmentLength + operand2SDSegmentLength + operand2SegmentLength + resultSegmentLength > instrBitwidth - instrTypeBitwidth - instrOpcodeWidth)
    {
        out.fatal(CALL_INFO, -1, "Invalid instruction format\n");
    }

    // Extract segments
    uint32_t mode = (content & ((1 << modeSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength);
    uint32_t operand1 = (content & ((1 << operand1SegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength - operand1SegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength - operand1SegmentLength);
    uint32_t operand2SD = (content & ((1 << operand2SDSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength - operand1SegmentLength - operand2SDSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength - operand1SegmentLength - operand2SDSegmentLength);
    uint32_t operand2 = (content & ((1 << operand2SegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength - operand1SegmentLength - operand2SDSegmentLength - operand2SegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - modeSegmentLength - operand1SegmentLength - operand2SDSegmentLength - operand2SegmentLength);
    uint32_t result = (content & ((1 << resultSegmentLength) - 1));

    std::string operationStr;

    switch (mode)
    {
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
        boolRegisters[result] = boolRegisters[operand1] && boolRegisters[operand2]; // TODO check this
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

    out.output("CALCULATE: mode=%s (%s), operand1=%s, operand2SD=%s, operand2=%s, result=%s\n", std::bitset<6>(mode).to_string().c_str(), operationStr.c_str(), std::bitset<4>(operand1).to_string().c_str(), std::bitset<1>(operand2SD).to_string().c_str(), std::bitset<8>(operand2).to_string().c_str(), std::bitset<4>(result).to_string().c_str());
}

void Sequencer::branch(uint32_t content)
{
    uint32_t regSegmentLength = 4;
    int32_t targetTrueSegmentLength = 9;
    int32_t targetFalseSegmentLength = 9;

    // Check validity
    if (regSegmentLength + targetTrueSegmentLength + targetFalseSegmentLength > instrBitwidth - instrTypeBitwidth - instrOpcodeWidth)
    {
        out.fatal(CALL_INFO, -1, "Invalid instruction format\n");
    }

    // Extract segments
    uint32_t reg = (content & ((1 << regSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - regSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - regSegmentLength);
    int32_t targetTrue = (content & ((1 << targetTrueSegmentLength) - 1) << (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - regSegmentLength - targetTrueSegmentLength)) >> (instrBitwidth - instrTypeBitwidth - instrOpcodeWidth - regSegmentLength - targetTrueSegmentLength);
    int32_t targetFalse = (content & ((1 << targetFalseSegmentLength) - 1));

    // Sign extend targetTrue and targetFalse
    if (targetTrue & (1 << (targetTrueSegmentLength - 1)))
    {
        targetTrue |= ~((1 << targetTrueSegmentLength) - 1);
    }
    if (targetFalse & (1 << (targetFalseSegmentLength - 1)))
    {
        targetFalse |= ~((1 << targetFalseSegmentLength) - 1);
    }

    // Compute new PC
    if (boolRegisters[reg])
    {
        pc += targetTrue;
    }
    else
    {
        pc += targetFalse;
    }

    out.output("BRANCH: reg=%s, targetTrue=%d, targetFalse=%d\n", std::bitset<4>(reg).to_string().c_str(), targetTrue, targetFalse);
}