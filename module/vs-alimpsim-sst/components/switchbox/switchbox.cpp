#include "switchbox.h"

#include <bitset>
#include <sst/core/component.h>
#include <sst/core/link.h>

#include "dataEvent.h"

Switchbox::Switchbox(ComponentId_t id, Params &params)
    : DRRAResource(id, params) {
  // Number of FSMs
  numFSMs = params.find<uint32_t>("number_of_fsms", 4);
  for (uint32_t i = 0; i < numFSMs; i++) {
    connection_maps.push_back(map<uint32_t, uint32_t>());
    sending_routes_maps.push_back(map<uint32_t, vector<uint32_t>>());
    receiving_routes_maps.push_back(map<uint32_t, vector<uint32_t>>());
  }
  // Set FSM 0 as the default FSM
  switchToFSM(0);

  // Slot ports
  num_slots = params.find<uint32_t>("num_slots", 16);
  slot_links.reserve(num_slots);
  std::string linkName;
  std::vector<uint32_t> connected_links;
  for (uint32_t swb_slot_conn_id = 0; swb_slot_conn_id < num_slots;
       swb_slot_conn_id++) {
    linkName = "slot_port" + std::to_string(swb_slot_conn_id);
    if (isPortConnected(linkName)) {
      if (swb_slot_conn_id == slot_id) {
        out.fatal(CALL_INFO, -1,
                  "SWB cannot be connected to itself (slot %d)\n", slot_id);
      }

      Link *link = configureLink(
          linkName, "0ns",
          new Event::Handler2<Switchbox, &Switchbox::handleSlotEventWithID,
                              uint32_t>(this, swb_slot_conn_id));
      sst_assert(link, CALL_INFO, -1, "Failed to configure link %s\n",
                 linkName.c_str());

      slot_links[swb_slot_conn_id] = link;
      connected_links.push_back(swb_slot_conn_id);
    }
  }
  out.output("Connected %lu slot links (", connected_links.size());
  for (auto link : connected_links) {
    out.print("%u,", link);
  }
  out.print(")\n");

  // Cell ports
  std::vector<uint32_t> totalConnections;
  cell_links.reserve(9); // 9 cell ports
  for (uint8_t dir = CellDirection::NW; dir <= CellDirection::SE; dir++) {
    linkName = "cell_port" + std::to_string(dir);
    if (isPortConnected(linkName)) {
      Link *link = configureLink(
          linkName, "0ns",
          new Event::Handler2<Switchbox, &Switchbox::handleCellEventWithID,
                              uint32_t>(this, dir));

      sst_assert(link, CALL_INFO, -1, "Failed to configure link %s\n",
                 linkName.c_str());

      cell_links[dir] = link;
      totalConnections.push_back(dir);
    }
  }
  out.output("Connected %u cell links (", totalConnections.size());
  for (auto link : totalConnections) {
    out.print("%s,", cell_directions_str[link].c_str());
    // out.print("%u,", link);
  }
  out.print(")\n");

  registerClock(
      clock, new SST::Clock::Handler<Switchbox>(this, &Switchbox::clockTick));
}

Switchbox::~Switchbox() {}

void Switchbox::init(unsigned int phase) {
  out.verbose(CALL_INFO, 1, 0, "Initialized\n");
}

void Switchbox::setup() { currentEventNumber++; }

void Switchbox::complete(unsigned int phase) {}

void Switchbox::finish() { out.verbose(CALL_INFO, 1, 0, "Finishing\n"); }

bool Switchbox::clockTick(Cycle_t currentCycle) {
  executeScheduledEventsForCycle(currentCycle);
  return false;
}

void Switchbox::handleSlotEventWithID(Event *event, uint32_t id) {
  DataEvent *dataEvent = dynamic_cast<DataEvent *>(event);
  if (dataEvent) {
    // if (!active_ports[dataEvent->portType]) {
    //   out.fatal(CALL_INFO, -1, "Received data while inactive\n", slot_id);
    // }

    // Verify if the slot is mapped to another slot
    if (connection_maps[currentFsmPort].count(id)) {
      uint32_t target = connection_maps[currentFsmPort][id];
      out.output("Forwarding data from slot %u to slot %u\n", id, target);
      slot_links[target]->send(event);
    } else if (sending_routes_maps[currentFsmPort].count(id)) {
      for (auto target : sending_routes_maps[currentFsmPort][id]) {
        if (cell_links[target] == nullptr) {
          out.fatal(CALL_INFO, -1, "Cell link %u is not linked\n", id);
        }
        out.output("Forwarding data to adjacent cell (direction: %s)\n",
                   cell_directions_str[target].c_str());
        cell_links[target]->send(event);
      }
    } else {
      for (uint32_t fsm_id = 0; fsm_id < numFSMs; fsm_id++) {
        out.output("Current FSM: %u\n", currentFsmPort);
        out.output("FSM %u connections: %lu\n", fsm_id,
                   connection_maps[fsm_id].size());
        for (const auto &conn : connection_maps[fsm_id]) {
          out.output("FSM %u connection from slot %u to slot %u\n", fsm_id,
                     conn.first, conn.second);
        }
        out.output("FSM %u receiving routes: %lu\n", fsm_id,
                   receiving_routes_maps[fsm_id].size());
        for (const auto &route : receiving_routes_maps[fsm_id]) {
          out.output("FSM %u receiving route from cell %u to slots ", fsm_id,
                     route.first);
          for (const auto &slot : route.second) {
            out.print("%u,", slot);
          }
          out.print("\n");
        }
        out.output("FSM %u sending routes: %lu\n", fsm_id,
                   sending_routes_maps[fsm_id].size());
        for (const auto &route : sending_routes_maps[fsm_id]) {
          out.output("FSM %u sending route from slot %u to cells ", fsm_id,
                     route.first);
          for (const auto &cell : route.second) {
            out.print("%s,", cell_directions_str[cell].c_str());
          }
          out.print("\n");
        }
      }

      out.fatal(CALL_INFO, -1, "Slot %u is not linked\n", id);
    }
  }
}

void Switchbox::handleCellEventWithID(Event *event, uint32_t id) {
  DataEvent *dataEvent = dynamic_cast<DataEvent *>(event);
  if (dataEvent)
    out.output("Received data from adjacent cell (direction: %s)\n",
               cell_directions_str[id].c_str());

  // Verify if the slot is mapped to another slot
  if (receiving_routes_maps[currentFsmPort].count(id)) {
    if (receiving_routes_maps[currentFsmPort][id].size() > 1) {
      out.output("Broadcasting data from cell %s to slots ",
                 cell_directions_str[id].c_str());
    } else {
      out.output("Forwarding from cell %s data to slot ",
                 cell_directions_str[id].c_str());
    }
    for (int i = 0; i < receiving_routes_maps[currentFsmPort][id].size(); i++) {
      uint32_t target = receiving_routes_maps[currentFsmPort][id][i];
      out.print("%u", target);
      if (i < receiving_routes_maps[currentFsmPort][id].size() - 1) {
        out.print(", ");
      }
      slot_links[target]->send(event);
    }
    out.print("\n");
  } else {
    out.fatal(CALL_INFO, -1, "Cell %u is not linked\n", id);
  }
}

void Switchbox::decodeInstr(uint32_t instr) {
  // Decode instruction
  uint32_t opcode = getInstrOpcode(instr);
  switch (opcode) {
  case REP: // repetition instruction
    handleRep(instr);
    break;
  case REPX: // repetition instruction
    handleRepx(instr);
    break;
  case FSM: // transition instruction
    // pendingFSMInstr = instr; // save FSM instruction
    handleFsm(instr);
    break;
  case SWB: // event instruction
    // Switch FSM port if FSM instruction is pending
    // if (pendingFSMInstr) {
    //   handleFsm(pendingFSMInstr);
    // }
    // Add handler to the current FSM port
    // fsmHandlers["fsm_" + std::to_string(currentFsmPort)].push_back(
    //     [this, instr] { handleSwb(instr); });
    handleSwb(instr);
    break;
  case ROUTE: // event instruction
    // Add handler to the current FSM port
    // fsmHandlers["fsm_" + std::to_string(currentFsmPort)].push_back(
    //     [this, instr] { handleRoute(instr); });
    handleRoute(instr);
    break;
  default:
    out.fatal(CALL_INFO, -1, "Invalid opcode\n");
    break;
  }
}

void Switchbox::handleRep(uint32_t instr) {
  // Instruction fields
  uint32_t port = getInstrField(instr, 2, 22);
  uint32_t level = getInstrField(instr, 4, 18);
  uint32_t iter = getInstrField(instr, 6, 12);
  uint32_t step = getInstrField(instr, 6, 6);
  uint32_t delay = getInstrField(instr, 6, 0);

  // For now, we only support increasing repetition levels (and no skipping)
  if (level != lastRepLevel + 1) {
    out.fatal(CALL_INFO, -1, "Invalid repetition level (last=%u, curr=%u)\n",
              lastRepLevel, level);
  } else {
    lastRepLevel = level;
  }

  // add repetition to the timing model
  try {
    next_timing_states[port].addRepetition(iter, delay, level, step);
  } catch (const std::exception &e) {
    out.fatal(CALL_INFO, -1, "Failed to add repetition: %s\n", e.what());
  }
}

void Switchbox::handleRepx(uint32_t instr) { handleRep(instr); }

void Switchbox::handleFsm(uint32_t instr) {
  // Instruction fields
  uint32_t port = getInstrField(instr, 3, 21);
  uint32_t delay_0 = getInstrField(instr, 7, 14);
  uint32_t delay_1 = getInstrField(instr, 7, 7);
  uint32_t delay_2 = getInstrField(instr, 7, 0);
  // TODO: what are the use cases for delay_1 and delay_2?

  // add transition to the timing model
  try {
    next_timing_states[port].addTransition(
        delay_0, "event_" + std::to_string(currentEventNumber),
        [this, port] { switchToFSM(port); });
    currentEventNumber++;
  } catch (const std::exception &e) {
    out.fatal(CALL_INFO, -1, "Failed to add transition: %s\n", e.what());
  }
}

void Switchbox::handleSwb(uint32_t instr) {
  // Instruction fields
  uint32_t option = getInstrField(instr, 2, 22);
  uint32_t channel = getInstrField(instr, 4, 18);
  uint32_t source = getInstrField(instr, 4, 14);
  uint32_t target = getInstrField(instr, 4, 10);

  if (channel != target) {
    out.fatal(CALL_INFO, -1,
              "Invalid channel\nSWB implemented as a "
              "crossbar\n");
  }

  // Add the connection to the SWB map
  connection_maps[option][source] = target;

  out.output("Adding connection from slot %u to slot %u "
             "in FSM %u\n",
             source, target, option);
}

void Switchbox::handleRoute(uint32_t instr) {
  // Instruction fields
  uint32_t option = getInstrField(instr, 2, 22);
  bool is_receive = getInstrField(instr, 1, 21) == 1;
  uint32_t source = getInstrField(instr, 4, 17);
  uint32_t target = getInstrField(instr, 16, 1);
  vector<uint32_t> targets;

  out.output("route (slot=%u, option=%u, sr=%u, source=%u, target=%s)\n",
             getInstrSlot(instr), option, is_receive, source,
             bitset<16>(target).to_string().c_str());

  if (is_receive) {
    // Receive
    // source is cell (NW=0/N/NE/W/C/E/SW/S/SE)
    // target is slot number (1-hot encoded)

    // Convert 1-hot encoded target to slot number
    for (uint32_t i = 0; i < 16; i++) {
      if (target & (1 << i)) {
        targets.push_back(i);
        receiving_routes_maps[option][source].push_back(i);
      }
    }
  } else {
    // Send
    // source is slot number
    // target is cell (NW=0/N/NE/W/C/E/SW/S/SE) (1-hot encoded)

    // Convert 1-hot encoded target to cell number
    for (uint32_t i = 0; i < 16; i++) {
      if (target & (1 << i)) {
        targets.push_back(i);
        sending_routes_maps[option][source].push_back(i);
      }
    }
  }

  out.output("Adding %s route from %s to [",
             is_receive ? "receiving" : "sending",
             is_receive ? cell_directions_str[source].c_str()
                        : to_string(source).c_str());
  for (size_t i = 0; i < targets.size(); ++i) {
    if (is_receive) {
      out.print("%u", targets[i]);
    } else {
      out.print("%s", cell_directions_str[targets[i]].c_str());
    }
    if (i < targets.size() - 1) {
      out.print(", ");
    }
  }
  out.print("] in FSM %u\n", option);
}
