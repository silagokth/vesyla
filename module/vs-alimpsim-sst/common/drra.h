#ifndef _DRRA_H
#define _DRRA_H

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>

#include "timingModel.h"

using namespace SST;

class DRRAOutput : public Output {
private:
  std::string prefix;

public:
  DRRAOutput(const std::string &prefix = "") : prefix(prefix) {}

  void setPrefix(const std::string &new_prefix) { prefix = new_prefix; }

  template <typename... Args> void output(const char *format, Args... args) {
    std::string prefixed_format = prefix + format;
    if constexpr (sizeof...(args) == 0) {
      Output::output("%s", prefixed_format.c_str());
    } else {
      Output::output(prefixed_format.c_str(), args...);
    }
  }

  template <typename... Args>
  void fatal(uint32_t line, const char *file, const char *func, int exit_code,
             const char *format, Args... args) {
    std::string prefixed_format = prefix + format;
    if constexpr (sizeof...(args) == 0) {
      Output::fatal(line, file, func, exit_code, "%s", prefixed_format.c_str());
    } else {
      Output::fatal(line, file, func, exit_code, prefixed_format.c_str(),
                    args...);
    }
  }

  template <typename... Args>
  void verbose(uint32_t line, const char *file, const char *func,
               uint32_t output_level, uint32_t output_bits, const char *format,
               Args... args) {
    std::string prefixed_format = prefix + format;
    if constexpr (sizeof...(args) == 0) {
      Output::verbose(line, file, func, output_level, output_bits, "%s",
                      prefixed_format.c_str());
    } else {
      Output::verbose(line, file, func, output_level, output_bits,
                      prefixed_format.c_str(), args...);
    }
  }

  template <typename... Args> void print(const char *format, Args... args) {
    if constexpr (sizeof...(args) == 0) {
      Output::output("%s", format);
    } else {
      Output::output(format, args...);
    }
  }
};

class DRRAComponent : public Component {
public:
  DRRAComponent(ComponentId_t id, Params &params) : Component(id) {
    // Configure init
    out.init("", 16, 0, Output::STDOUT);

    // Get parameters
    clock = params.find<std::string>("clock", "100MHz");
    printFrequency = params.find<Cycle_t>("printFrequency", 1);
    word_bitwidth = params.find<size_t>("word_bitwidth", 16);

    // Cell coordinates
    std::vector<int> paramsCellCoordinates;
    params.find_array<int>("cell_coordinates", paramsCellCoordinates);
    if (paramsCellCoordinates.size() != 2) {
      out.output("Size of cell coordinates: %lu\n",
                 paramsCellCoordinates.size());
      out.fatal(CALL_INFO, -1, "Invalid cell coordinates\n");
    } else {
      cell_coordinates[0] = paramsCellCoordinates[0];
      cell_coordinates[1] = paramsCellCoordinates[1];
    }

    // Instruction format
    instrBitwidth = params.find<uint64_t>("instr_bitwidth", 32);
    instrTypeBitwidth = params.find<uint64_t>("instr_type_bitwidth", 1);
    instrOpcodeWidth = params.find<uint64_t>("instr_opcode_width", 3);
    instrSlotWidth = params.find<uint64_t>("instr_slot_width", 4);
  }

  virtual ~DRRAComponent() {}

  // SST lifecycle methods
  virtual void init(unsigned int phase) override = 0;
  virtual void setup() override = 0;
  virtual void complete(unsigned int phase) override = 0;
  virtual void finish() override = 0;

  // SST clock handler
  virtual bool clockTick(Cycle_t currentCycle) = 0;

  // SST event handler
  virtual void handleEvent(Event *event) = 0;

protected:
  // Document params
  static std::vector<SST::ElementInfoParam> getBaseParams() {
    std::vector<SST::ElementInfoParam> params;
    params.push_back({"clock", "Clock frequency", "100MHz"});
    params.push_back(
        {"printFrequency", "Frequency to print tick messages", "1000"});
    params.push_back({"io_data_width", "Width of the IO data", "256"});
    params.push_back({"word_bitwidth", "Width of the word", "16"});
    params.push_back({"instr_bitwidth", "Instruction bitwidth", "32"});
    params.push_back({"instr_type_bitwidth", "Instruction type", "1"});
    params.push_back({"instr_opcode_width", "Instruction opcode width", "3"});
    params.push_back({"instr_slot_width", "Instruction slot width", "4"});
    params.push_back({"cell_coordinates", "Cell coordinates", "[0,0]"});
    return params;
  }

  // Simulation global variables
  DRRAOutput out;
  std::string clock;
  Cycle_t printFrequency;

  // Local buffers
  uint32_t instrBuffer;

  // Cell coordinates
  uint32_t cell_coordinates[2] = {0, 0};

  // // Links (controller and data)
  // Link *controller_link;
  // Link *data_link;

  // Narrow level
  size_t word_bitwidth;

  // Instruction format (from isa.json file)
  uint32_t instrBitwidth;
  uint32_t instrTypeBitwidth;
  uint32_t instrOpcodeWidth;
  uint32_t instrSlotWidth;

  uint32_t getInstrType(uint32_t instr) {
    return getInstrField(instr, instrTypeBitwidth,
                         instrBitwidth - instrTypeBitwidth);
  }

  uint32_t getInstrOpcode(uint32_t instr) {
    return getInstrField(instr, instrOpcodeWidth,
                         instrBitwidth - instrTypeBitwidth - instrOpcodeWidth);
  }

  uint32_t getInstrSlot(uint32_t instr) {
    return getInstrField(instr, instrSlotWidth,
                         instrBitwidth - instrTypeBitwidth - instrOpcodeWidth -
                             instrSlotWidth);
  }

  uint32_t isResourceInstruction(uint32_t instr) {
    return getInstrType(instr) == 1;
  }

  uint32_t isControlInstruction(uint32_t instr) {
    return getInstrType(instr) == 0;
  }

  uint32_t getInstrField(uint32_t instr, uint32_t fieldWidth,
                         uint32_t fieldOffset) {
    // Validate field parameters against instruction width
    sst_assert(fieldWidth > 0, CALL_INFO, -1, "Field width must be positive");
    sst_assert(fieldOffset + fieldWidth <= instrBitwidth, CALL_INFO, -1,
               "Field (width=%u, offset=%u) exceeds instruction width %u",
               fieldWidth, fieldOffset, instrBitwidth);

    return (instr & ((1 << fieldWidth) - 1) << fieldOffset) >> fieldOffset;
  }

  std::string formatRawDataToWords(std::vector<uint8_t> raw_data) {
    std::string formatted_data = "[";
    size_t word_size = word_bitwidth / 8;
    uint64_t current_word = 0;
    for (size_t i = 0; i < raw_data.size(); i++) {
      current_word |= raw_data[i] << (i % word_size * 8);
      if ((i + 1) % word_size == 0) {
        formatted_data += std::to_string(current_word);
        current_word = 0;
        if (i + 1 < raw_data.size()) {
          formatted_data += ", ";
        }
      }
    }
    formatted_data += "]";
    return formatted_data;
  }
};

class DRRAResource : public DRRAComponent {
public:
  DRRAResource(ComponentId_t id, Params &params) : DRRAComponent(id, params) {
    // Resource-specific params
    slot_id = params.find<int16_t>("slot_id", -1);
    io_data_width = params.find<uint32_t>("io_data_width", 256);

    // Configure output
    out.setPrefix(getType() + " [" + std::to_string(cell_coordinates[0]) + "_" +
                  std::to_string(cell_coordinates[1]) + "_" +
                  std::to_string(slot_id) + "] - ");

    has_io_input_connection =
        params.find<bool>("has_io_input_connection", false);
    has_io_output_connection =
        params.find<bool>("has_io_output_connection", false);

    // Resource size
    resource_size = params.find<uint8_t>("resource_size", 1);
    for (uint8_t i = 0; i < resource_size; i++) {
      for (uint8_t j = 0; j < 4; j++) {
        active_ports[i * 4 + j] = false;
        active_ports_cycles[i * 4 + j] = 0;
        next_timing_states[i * 4 + j] = TimingState();
        next_timing_states[i * 4 + j].addEvent(
            "event_0", [this] { out.output(" FSM switched to 0\n"); });
        port_last_rep_level[i * 4 + j] = -1;
      }
    }

    // Annotate slot IDs
    slot_ids.reserve(resource_size);
    for (uint8_t i = 0; i < resource_size; i++) {
      slot_ids.push_back(slot_id + i);
    }

    // Configure links
    for (uint8_t i = 0; i < resource_size; i++) {
      if (isPortConnected("controller_port" + std::to_string(i))) {
        controller_links.push_back(
            configureLink("controller_port" + std::to_string(i),
                          new Event::Handler<DRRAResource>(
                              this, &DRRAResource::handleEvent)));
      } else {
        controller_links.push_back(nullptr);
      }

      if (isPortConnected("data_port" + std::to_string(i))) {
        data_links.push_back(
            configureLink("data_port" + std::to_string(i),
                          new Event::Handler<DRRAResource>(
                              this, &DRRAResource::handleEvent)));
      } else {
        data_links.push_back(nullptr);
      }
    }
    sst_assert(controller_links.size() == resource_size, CALL_INFO, -1,
               "Controller links size mismatch");
    sst_assert(data_links.size() == resource_size, CALL_INFO, -1,
               "Data links size mismatch");
  }

protected:
  static std::vector<SST::ElementInfoParam> getBaseParams() {
    std::vector<SST::ElementInfoParam> params = DRRAComponent::getBaseParams();
    params.push_back({"slot_id",
                      "Slot ID for the resource. If the resource occupies "
                      "multiple slots, slot ID of the first slot occupied.",
                      "-1"});
    params.push_back(
        {"resource_size",
         "Number of slots occupied by the resource (default to 1 slot)", "1"});
    params.push_back(
        {"has_io_input_connection", "Has IO input connection", "0"});
    params.push_back(
        {"has_io_output_connection", "Has IO output connection", "0"});
    return params;
  }

  static std::vector<SST::ElementInfoPort> getBasePorts() {
    std::vector<SST::ElementInfoPort> ports;
    ports.push_back(
        {"controller_port%(portnum)d", "Link to the controller port", {""}});
    ports.push_back({"data_port%(portnum)d", "Link to the data port", {""}});
    ports.push_back({"io_input_port", "Link to the IO input port", {""}});
    ports.push_back({"io_output_port", "Link to the IO output port", {""}});
    return ports;
  }

  bool isPortActive(uint32_t port) { return active_ports[port]; }

  void activatePort(uint32_t port) {
    active_ports[port] = true;
    current_timing_states[port] = next_timing_states[port];
    next_timing_states[port] = TimingState();
    current_timing_states[port].build();
    port_last_rep_level[port] = -1;
  }

  // void activatePorts(uint32_t ports) {
  //   for (uint32_t i = 0; i < resource_size; i++) {
  //     if ((ports & (1 << i)) >> i) {
  //       activatePort(i);
  //     }
  //   }
  // }

  void activatePortsForSlot(uint32_t slot_id, uint32_t ports) {
    uint8_t slot_pos = std::distance(
        slot_ids.begin(), std::find(slot_ids.begin(), slot_ids.end(), slot_id));
    for (uint8_t i = 0; i < 4; i++) {
      if ((ports & (1 << i)) >> i) {
        activatePort(slot_pos * 4 + i);
      }
    }
  }

  uint32_t getPortActiveCycle(uint32_t port) {
    return active_ports_cycles[port];
  }

  auto getPortEventsForCycle(uint32_t port, uint32_t cycle) {
    return current_timing_states[port].getEventsForCycle(cycle);
  }

  // void resetTimingStates() {
  //   for (uint8_t i = 0; i < resource_size * 4; i++) {
  //     timing_states[i] = TimingState();
  //   }
  // }

  // Links
  std::vector<Link *> controller_links;
  std::vector<Link *> data_links;

  // Activation
  std::map<uint32_t, bool> active_ports;
  std::map<uint32_t, uint32_t> active_ports_cycles;

  // IO settings
  uint32_t io_data_width; // in bits
  bool has_io_input_connection, has_io_output_connection;

  // Slot settings
  int16_t slot_id;
  std::vector<int16_t> slot_ids;
  uint8_t resource_size = 1; // Default to 1 slot

  // Timing model state for each ports
  std::map<uint32_t, TimingState> current_timing_states;
  std::map<uint32_t, TimingState> next_timing_states;
  // std::map<std::string, std::function<void()>> events_handlers_map;
  std::map<uint32_t, int32_t> port_last_rep_level;
};

class DRRAController : public DRRAComponent {
public:
  DRRAController(ComponentId_t id, Params &params) : DRRAComponent(id, params) {
    // Configure output
    out.setPrefix(getType() + " [" + std::to_string(cell_coordinates[0]) + "_" +
                  std::to_string(cell_coordinates[1]) + "] - ");

    // Get number of slots
    num_slots = params.find<uint32_t>("num_slots", 16);

    // Configure slot links
    slot_links.reserve(num_slots);
    uint8_t num_connected_links = 0;
    for (uint32_t i = 0; i < num_slots; i++) {
      if (isPortConnected("slot_port" + std::to_string(i))) {
        slot_links.push_back(
            configureLink("slot_port" + std::to_string(i),
                          new Event::Handler<DRRAController>(
                              this, &DRRAController::handleEvent)));
        num_connected_links++;
      } else {
        slot_links.push_back(nullptr);
      }
    }
    sst_assert(slot_links.size() == num_slots, CALL_INFO, -1,
               "Slot links size mismatch");
    if (num_connected_links > 0) {
      out.output("Connected %lu slot links (", num_connected_links);
      for (uint32_t i = 0; i < num_slots; i++) {
        if (slot_links[i] != nullptr) {
          out.print("%u,", i);
        }
      }
      out.print(")\n");
    }
  }

  static std::vector<SST::ElementInfoParam> getBaseParams() {
    std::vector<SST::ElementInfoParam> params = DRRAComponent::getBaseParams();
    params.push_back({"num_slots",
                      "Number of slots that can be connected to the controller",
                      "16"});
    return params;
  }

  static std::vector<SST::ElementInfoPort> getBasePorts() {
    std::vector<SST::ElementInfoPort> ports;
    ports.push_back({"slot_port%(portnum)d",
                     "Link(s) to resources in slots. Connect slot_port0, "
                     "slot_port1, etc."});
    return ports;
  }

protected:
  // Program counter
  uint32_t pc = 0;

  // Slot links
  uint32_t num_slots;
  std::vector<Link *> slot_links;
};

#endif // _DRRA_H
