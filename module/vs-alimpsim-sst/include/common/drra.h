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
    printFrequency = params.find<Cycle_t>("printFrequency", 1000);
    io_data_width = params.find<uint32_t>("io_data_width", 256);

    slot_id = params.find<int16_t>("slot_id", -1);
    has_io_input_connection =
        params.find<bool>("has_io_input_connection", false);
    has_io_output_connection =
        params.find<bool>("has_io_output_connection", false);

    // Cell coordinates
    std::vector<int> paramsCellCoordinates;
    params.find_array<int>("cell_coordinates", paramsCellCoordinates);
    if (paramsCellCoordinates.size() != 2) {
      out.output("Size of cell coordinates: %lu\n",
                 paramsCellCoordinates.size());
      out.fatal(CALL_INFO, -1, "Invalid cell coordinates\n");
    } else {
      cellCoordinates[0] = paramsCellCoordinates[0];
      cellCoordinates[1] = paramsCellCoordinates[1];
    }

    // Instruction format
    instrBitwidth = params.find<uint64_t>("instr_bitwidth", 32);
    instrTypeBitwidth = params.find<uint64_t>("instr_type_bitwidth", 1);
    instrOpcodeWidth = params.find<uint64_t>("instr_opcode_width", 3);
    instrSlotWidth = params.find<uint64_t>("instr_slot_width", 4);

    // Configure output
    if (slot_id != -1) {
      out.setPrefix(getType() + " [" + std::to_string(cellCoordinates[0]) +
                    "_" + std::to_string(cellCoordinates[1]) + "_" +
                    std::to_string(slot_id) + "] - ");
    } else {
      out.setPrefix(getType() + " [" + std::to_string(cellCoordinates[0]) +
                    "_" + std::to_string(cellCoordinates[1]) + "] - ");
    }
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
    params.push_back({"slot_id", "Slot ID", "-1"});
    params.push_back({"instr_bitwidth", "Instruction bitwidth", "32"});
    params.push_back({"instr_type_bitwidth", "Instruction type", "1"});
    params.push_back({"instr_opcode_width", "Instruction opcode width", "3"});
    params.push_back({"instr_slot_width", "Instruction slot width", "4"});
    params.push_back(
        {"has_io_input_connection", "Has IO input connection", "0"});
    params.push_back(
        {"has_io_output_connection", "Has IO output connection", "0"});
    params.push_back({"cell_coordinates", "Cell coordinates", "[0,0]"});
    return params;
  }

  // Simulation global variables
  DRRAOutput out;
  std::string clock;
  Cycle_t printFrequency;

  // Local buffers
  bool isActive = false;
  uint32_t instrBuffer;

  // Slot & cell settings
  int16_t slot_id = -1;
  uint32_t cellCoordinates[2] = {0, 0};

  // Links (controller and data)
  Link *controllerLink;
  Link *dataLink;

  // IO settings
  uint32_t io_data_width;
  bool has_io_input_connection, has_io_output_connection;

  // Timing model state
  TimingState timingState = TimingState();

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
};

#endif // _DRRA_H
