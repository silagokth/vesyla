#ifndef _SWITCHBOX_H
#define _SWITCHBOX_H

#include "drra.h"

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/sst_types.h>
#include <sst/core/timeConverter.h>

using namespace std;
using namespace SST;

class Switchbox : public DRRAResource {
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(Switchbox,   // Class name
                             "drra",      // Name of library
                             "Switchbox", // Lookup name for component
                             SST_ELI_ELEMENT_VERSION(1, 0,
                                                     0),  // Component version
                             "Switchbox component",       // Description
                             COMPONENT_CATEGORY_PROCESSOR // Category
  )

  /* Element Library Params */
  static vector<ElementInfoParam> getComponentParams() {
    auto params = DRRAResource::getBaseParams();
    params.push_back({"number_of_fsms", "Number of FSMs", "4"});
    params.push_back({"num_slots", "Number of slots", "16"});
    return params;
  }
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  /* Element Library Ports */
  static vector<ElementInfoPort> getComponentPorts() {
    auto ports = DRRAResource::getBasePorts();
    ports.push_back(
        {"slot_port%(portnum)d",
         "Link(s) to resources slots (slot_port0, slot_port1, etc.)"});
    ports.push_back({"cell_port%(portnum)d",
                     "Link(s) to cells (cell_port0, cell_port1, etc.)"});
    ports.push_back(
        {"input_buffer_port", "Link to the input buffer (optional)"});
    ports.push_back(
        {"output_buffer_port", "Link to the output buffer (optional)"});
    return ports;
  }
  SST_ELI_DOCUMENT_PORTS(getComponentPorts())

  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()

  /* Constructor */
  Switchbox(ComponentId_t id, Params &params);

  /* Destructor */
  ~Switchbox();

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
  // Decode instruction
  void decodeInstr(uint32_t instr);

  void switchToFSM(uint32_t fsmPort) {
    currentFsmPort = fsmPort;
    out.output("Switching to FSM port %u\n", currentFsmPort);
  }

  // Different supported opcodes
  enum Opcode {
    REP,
    REPX,
    FSM,
    SWB = 4,
    ROUTE,
  };

  void handleRep(uint32_t instr);
  void handleRepx(uint32_t instr);
  void handleFsm(uint32_t instr);
  void handleSwb(uint32_t instr);
  void handleRoute(uint32_t instr);

  // Map input ports to output ports ([source] = target)
  vector<map<uint32_t, uint32_t>> connection_maps;
  vector<map<uint32_t, uint32_t>> sending_routes_maps;
  vector<map<uint32_t, uint32_t>> receiving_routes_maps;

  // SST links
  // Controller link
  Link *controller_link;

  // Slot links
  uint32_t num_slots;
  vector<Link *> slot_links;

  // Cell links
  vector<Link *> cell_links;

  // Cell directions
  enum CellDirection { NW, N, NE, W, C, E, SW, S, SE };
  string cell_directions_str[9] = {"NW", "N",  "NE", "W", "C",
                                   "E",  "SW", "S",  "SE"};

  // Handlers
  void handleSlotEventWithID(Event *event, uint32_t id);
  void handleCellEventWithID(Event *event, uint32_t id);

  // Events handlers list
  vector<function<void()>> eventsHandlers;

  uint32_t numFSMs = 4;
  uint32_t currentFsmPort = 0;
  int32_t lastRepLevel = -1;
  uint32_t currentEventNumber = 0;
  uint32_t pendingFSMInstr = 0;
  uint64_t activeCycle = 0;
};

#endif // _SWITCHBOX_H