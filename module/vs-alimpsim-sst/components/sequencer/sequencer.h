#ifndef _SEQUENCER_H
#define _SEQUENCER_H

#include <sst/core/component.h>
#include <sst/core/params.h>

using namespace std;
using namespace SST;

class Sequencer : public Component
{
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(
      Sequencer,                        // Class name
      "drra",                           // Name of library
      "Sequencer",                      // Lookup name for component
      SST_ELI_ELEMENT_VERSION(1, 0, 0), // Component version
      "Sequencer component",            // Description
      COMPONENT_CATEGORY_PROCESSOR      // Category
  )

  SST_ELI_DOCUMENT_PARAMS(
      {"clock", "Clock frequency", "100MHz"},
      {"print_frequency", "Frequency to print tick messages", "1000"},
      {"assembly_program_path", "Path to the assembly program file", ""},
      {"cell_coordinates", "Coordinates of the cell", "[0,0]"},

      // Params (from arch.json file)
      {"resource_instr_width", "Instruction length for this sequencer", "32"},       // RESOURCE_INSTR_WIDTH
      {"num_slots", "Number of slots that can be connected to the sequencer", "16"}, // NUM_SLOTS
      {"fsm_per_slot", "Number of FSM per slot"},                                    // FSM_PER_SLOT
      {"instr_data_width", "Instruction data width", "32"},                          // INSTR_DATA_WIDTH
      {"instr_addr_width", "Instruction address width", "6"},                        // INSTR_ADDR_WIDTH
      {"instr_hops_width", "Instruction hops width", "4"},                           // INSTR_HOPS_WIDTH

      // Instruction format (from isa.json file)
      {"instr_bitwidth", "Instruction bitwidth", "32"},
      {"instr_type_bitwidth", "Instruction type (control/resource)", "1"},
      {"instr_opcode_width", "Instruction opcode width", "3"},
      {"instr_slot_width", "Instruction slot width", "4"}, )

  SST_ELI_DOCUMENT_PORTS(
      {"slot_port%(portnum)d", "Link(s) to resources in slots. Connect slot_port0, slot_port1, etc."})
  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()

  /* Constructor */
  Sequencer(ComponentId_t id, Params &params);

  /* Destructor */
  ~Sequencer();

  // SST lifecycle methods
  virtual void init(unsigned int phase) override;
  virtual void setup() override;
  virtual void complete(unsigned int phase) override;
  virtual void finish() override;

  // SST clock handler
  bool clockTick(Cycle_t currentCycle);

  // SST event handler
  void handleEvent(Event *event);

private:
  Output out;
  std::string clock;
  Cycle_t printFrequency;
  bool readyToFinish = false;

  uint32_t cellCoordinates[2] = {0, 0};
  uint32_t cyclesToWait = 0;

  std::string assemblyProgramPath;
  std::vector<uint32_t> assemblyProgram;

  // Params (from arch.json file)
  uint32_t numSlots;
  uint32_t resourceInstrWidth;
  uint32_t fsmPerSlot;
  uint32_t instrDataWidth;
  uint32_t instrAddrWidth;
  uint32_t instrHopsWidth;

  // Instruction format (from isa.json file)
  uint32_t instrBitwidth;
  uint32_t instrTypeBitwidth;
  uint32_t instrOpcodeWidth;
  uint32_t instrSlotWidth;

  std::vector<Link *> slotLinks;

  uint32_t pc = 0;

  // Add scalar and bool registers
  std::vector<uint32_t> scalarRegisters;
  std::vector<bool> boolRegisters;

  // Add fetch_decode method
  void fetch_decode(uint32_t instruction);
  void load_assembly_program(std::string);

  // Add execute method
  void halt();
  void wait(uint32_t content);
  void wait_event();
  void wait_cycles(uint32_t cycles);
  void activate(uint32_t content);
  void calculate(uint32_t content);
  void branch(uint32_t content);
};

#endif // _SEQUENCER_H
