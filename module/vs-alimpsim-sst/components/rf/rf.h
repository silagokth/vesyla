#ifndef _RF_H
#define _RF_H

#include "dataEvent.h"
#include "drra.h"

using namespace std;
using namespace SST;

class RegisterFile : public DRRAResource {
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

  /* Element Library Params */
  static vector<ElementInfoParam> getComponentParams() {
    auto params = DRRAResource::getBaseParams();
    params.push_back({"access_time", "Time to access the IO buffer", "0ns"});
    params.push_back({"register_file_size",
                      "Size of the register file in number of words", "1024"});
    return params;
  }
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  /* Element Library Ports */
  static vector<ElementInfoPort> getComponentPorts() {
    auto ports = DRRAResource::getBasePorts();
    return ports;
  }
  SST_ELI_DOCUMENT_PORTS(getComponentPorts())
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

  enum PortMap {
    NarrowIn = 0,
    NarrowOut = 1,
    BulkIn = 2,
    BulkOut = 3,
  };

  // Decode instructions
  void decodeInstr(uint32_t instr);

  // Supported opcodes
  enum OpCode { REP = 0, REPX = 1, DSU = 6 };

  void handleRep(uint32_t instr);
  void handleRepx(uint32_t instr);
  void handleDSU(uint32_t instr);

  void sendWideData();
  void sendNarrowData();
  void receiveWideData();
  void receiveNarrowData();

  uint32_t current_event_number = 0;
  // int32_t lastRepLevel = -1;
  map<uint32_t, uint32_t> port_agus;
};

#endif // _RF_H