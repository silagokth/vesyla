#ifndef _VECADD_H
#define _VECADD_H

#include "drra.h"

#include <sst/core/component.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/params.h>

#include "ioEvents.h"

using namespace std;
using namespace SST;

class VecAdd : public DRRAResource {
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(VecAdd,   // Class name
                             "drra",   // Name of library
                             "VecAdd", // Lookup name for component
                             SST_ELI_ELEMENT_VERSION(1, 0,
                                                     0),  // Component version
                             "VecAdd DPU component",      // Description
                             COMPONENT_CATEGORY_PROCESSOR // Category
  )

  // Add component-specific parameters
  static vector<ElementInfoParam> getComponentParams() {
    auto params = DRRAResource::getBaseParams();
    params.push_back({"chunckWidth", "Width of the chunck", "16"});
    return params;
  }

  // Register the component parameters
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  SST_ELI_DOCUMENT_PORTS({"controller_port", "Link to the controller"},
                         {"input_buffer_port", "Link to the input buffer"},
                         {"output_buffer_port", "Link to the output buffer"})

  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()
  // {"input_buffer", "Input buffer", "SST::Interfaces::StandardMem"},
  // {"output_buffer", "Output buffer", "SST::Interfaces::StandardMem"})

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
  bool clockTick(Cycle_t currentCycle) override;

  // SST event handler
  void handleEvent(Event *event) override;
  void handleMemoryEvent(IOEvent *memEvent);

private:
  uint8_t chunckWidth;

  // Links
  Link *controllerLink =
      nullptr; // Sequencer connection (Instructions and activations)
  Link *inputBufferLink = nullptr;  // IO connection (data in)
  Link *outputBufferLink = nullptr; // IO connection (data out)

  // Memory interfaces
  // Interfaces::StandardMem *inputBuffer;  // IO connection (data in)
  // Interfaces::StandardMem *outputBuffer; // IO connection (data out)

  // Component state (IDLE, COMPUTE_0, COMPUTE_1)
  enum State { RESET, IDLE, COMPUTE_0, COMPUTE_1 };
  State state = RESET;

  // Add instruction (decoded)
  typedef struct vec_add_instr {
    bool en = false;
    uint16_t addr = 0;
  } VecAddInstr;

  VecAddInstr instr;
  std::vector<uint8_t> dataBuffer;

  // Functions
  VecAddInstr decodeInstruction(uint32_t instruction);
  void read_from_io();
  void write_to_io();
  void compute_addition();
};

#endif // _VECADD_H
