#ifndef _VECADD_H
#define _VECADD_H

#include <sst/core/component.h>
#include <sst/core/params.h>

using namespace std;
using namespace SST;

class VecAdd : public Component
{
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(
      VecAdd,                           // Class name
      "vec_add",                        // Name of library
      "VecAdd",                         // Lookup name for component
      SST_ELI_ELEMENT_VERSION(1, 0, 0), // Component version
      "VecAdd DPU component",           // Description
      COMPONENT_CATEGORY_PROCESSOR      // Category
  )

  SST_ELI_DOCUMENT_PARAMS(
      {"clock", "Clock frequency", "100MHz"},
      {"printFrequency", "Frequency to print tick messages", "1000"},
      {"chunckWidth", "Width of the chunck", "16"},
      {"io_data_width", "Width of the IO data", "256"},

      // Instruction format (from isa.json file)
      {"instr_bitwidth", "Instruction bitwidth", "32"},
      {"instr_type_bitwidth", "Instruction type (control/resource)", "1"},
      {"instr_opcode_width", "Instruction opcode width", "3"},
      {"instr_slot_width", "Instruction slot width", "4"}, )

  SST_ELI_DOCUMENT_PORTS(
      {"controller_port", "Link to the controller"})
  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()

  /* Constructor */
  VecAdd(ComponentId_t id, Params &params);

  /* Destructor */
  ~VecAdd();

  // SST lifecycle methods
  virtual void init(unsigned int phase) override;
  virtual void setup() override;
  virtual void complete(unsigned int phase) override;
  virtual void finish() override;

  // SST clock handler
  bool clockTick(Cycle_t currentCycle);

  // SST event handler
  void handleEvent(Event *event);
  void handleMemoryEvent(Interfaces::StandardMem::Request *req);

private:
  Output out;
  Cycle_t printFrequency;
  string clock;
  uint8_t chunckWidth = 16;

  uint32_t io_data_width = 256;

  // Instruction format (from isa.json file)
  uint32_t instrBitwidth;
  uint32_t instrTypeBitwidth;
  uint32_t instrOpcodeWidth;
  uint32_t instrSlotWidth;

  // Links
  Link *controllerLink; // Sequencer connection (Instructions and activations)
  // Component state (IDLE, COMPUTE_0, COMPUTE_1)
  enum State
  {
    RESET,
    IDLE,
    COMPUTE_0,
    COMPUTE_1
  };
  State state = RESET;

  // Add instruction (decoded)
  typedef struct vec_add_instr
  {
    bool en = false;
    uint16_t addr = 0;
  } VecAddInstr;

  bool active = false;
  uint32_t instrBuffer = 0;
  VecAddInstr instr;
  std::vector<uint8_t> dataBuffer;

  // Functions
  VecAddInstr decodeInstruction(uint32_t instruction);
  void read_from_io();
  void write_to_io();
  void compute_addition();
};

#endif // _VECADD_H
