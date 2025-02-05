#ifndef _RF_H
#define _RF_H

#include "dataEvent.h"
#include "drra.h"

using namespace std;
using namespace SST;

class RegisterFile : public DRRAComponent {
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(RegisterFile,   // Class name
                             "drra",         // Name of library
                             "RegisterFile", // Lookup name for component
                             SST_ELI_ELEMENT_VERSION(1, 0,
                                                     0), // Component version
                             "RegisterFile component",   // Description
                             COMPONENT_CATEGORY_MEMORY   // Category
  )

  // Add component-specific parameters
  static vector<ElementInfoParam> getComponentParams() {
    auto params = DRRAComponent::getBaseParams();
    params.push_back({"access_time", "Time to access the IO buffer", "0ns"});
    params.push_back({"register_file_size",
                      "Size of the register file in number of words", "1024"});
    return params;
  }

  // Register the component parameters
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  SST_ELI_DOCUMENT_PORTS(
      {"controller_port", "Link to the controller"}, // to receive instructions
      {"data_port", "Link to the switchbox"},        // to send and receive data
  )
  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()

  /* Constructor */
  RegisterFile(ComponentId_t id, Params &params);

  /* Destructor */
  ~RegisterFile();

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
  // Register File
  uint32_t register_file_size;
  std::string access_time;
  map<uint32_t, uint64_t> registers;

  // Decode instructions
  void decodeInstr(uint32_t instr);

  // Supported opcodes
  enum OpCode {
    REP = 0,
    REPX = 1,
    DSU = 6,
  };

  void handleRep(uint32_t instr);
  void handleRepx(uint32_t instr);
  void handleDSU(uint32_t instr);

  void sendWideData();
  void sendNarrowData();
  void receiveWideData();
  void receiveNarrowData();

  // Events handlers list
  vector<function<void()>> eventsHandlers;

  uint32_t current_event_number = 0;
  int32_t lastRepLevel = -1;
  int32_t agu_initial_addr = -1;
  uint64_t activeCycle = 0;
  uint32_t current_dsu_mode = DataEvent::PortType::WriteNarrow;
};

#endif // _RF_H