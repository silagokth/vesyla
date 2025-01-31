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

class Switchbox : public DRRAComponent {
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

  // Add component-specific parameters
  static vector<ElementInfoParam> getComponentParams() {
    auto params = DRRAComponent::getBaseParams();
    params.push_back({"number_of_fsms", "Number of FSMs", "4"});
    return params;
  }

  // Register the component parameters
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  SST_ELI_DOCUMENT_PORTS(
      {"controller_port", "Link to the controller"}, // to receive instructions
      {"slot_port%(portnum)d",
       "Link(s) to resources slots (slot_port0, slot_port1, etc.)"},
      // {"slot_output_port%(portnum)d", "Link(s) to resources slots
      // (output_port0, output_port1, etc.)"},
      {"cell_port%(portnum)d",
       "Link(s) to cells (cell_port0, cell_port1, etc.)"})

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
  map<uint32_t, uint32_t> connections;
  map<uint32_t, uint32_t> sending_routes;
  map<uint32_t, uint32_t> receiving_routes;

  // SST links
  // Controller link
  Link *controller_link;

  // Slot links
  vector<Link *> slot_links;

  // Cell links
  array<Link *, 9> cell_links;

  // Cell directions
  enum CellDirection { NW, N, NE, W, C, E, SW, S, SE };

  // Handlers
  void handleSlotEventWithID(Event *event, uint32_t id);
  void handleCellEventWithID(Event *event, uint32_t id);

  // Events handlers map
  vector<function<void()>> eventsHandlers;

  uint32_t numFSMs = 4;
  int32_t lastRepLevel = -1;
  uint32_t currentEventNumber = 0;
  uint32_t pendingFSMInstr = 0;
  uint64_t activeCycle = 0;
};

#endif // _SWITCHBOX_H