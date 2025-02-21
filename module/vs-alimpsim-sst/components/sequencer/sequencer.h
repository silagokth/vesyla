#ifndef _SEQUENCER_H
#define _SEQUENCER_H

#include "drra.h"

#include <sst/core/component.h>
#include <sst/core/params.h>

using namespace std;
using namespace SST;

class Sequencer : public DRRAController {
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(Sequencer,   // Class name
                             "drra",      // Name of library
                             "Sequencer", // Lookup name for component
                             SST_ELI_ELEMENT_VERSION(1, 0,
                                                     0),  // Component version
                             "Sequencer component",       // Description
                             COMPONENT_CATEGORY_PROCESSOR // Category
  )

  // Add component-specific parameters
  static vector<ElementInfoParam> getComponentParams() {
    auto params = DRRAController::getBaseParams();
    params.push_back(
        {"assembly_program_path", "Path to the assembly program file", ""});
    params.push_back({"fsm_per_slot", "Number of FSM per slot"});
    params.push_back({"instr_addr_width", "Instruction address width", "6"});
    params.push_back({"instr_hops_width", "Instruction hops width", "4"});
    return params;
  }
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  static vector<ElementInfoPort> getControllerPorts() {
    auto ports = DRRAController::getBasePorts();
    return ports;
  }
  SST_ELI_DOCUMENT_PORTS(getControllerPorts())
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
  bool clockTick(Cycle_t currentCycle) override;

  // SST event handler
  void handleEvent(Event *event) override;

private:
  // Output out;
  // std::string clock;
  // Cycle_t printFrequency;
  bool readyToFinish = false;

  // uint32_t cell_coordinates[2] = {0, 0};
  uint32_t cyclesToWait = 0;

  std::string assemblyProgramPath;
  std::vector<uint32_t> assemblyProgram;

  // Params (from arch.json file)
  uint32_t resourceInstrWidth;
  uint32_t fsmPerSlot;
  uint32_t instrDataWidth;
  uint32_t instrAddrWidth;
  uint32_t instrHopsWidth;

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
