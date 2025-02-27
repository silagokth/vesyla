#ifndef _DPU_H
#define _DPU_H

#include "drra.h"
#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/sst_types.h>
#include <sst/core/timeConverter.h>

class DPU : public DRRAResource {
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(DPU, "drra", "DPU",
                             SST_ELI_ELEMENT_VERSION(1, 0, 0), "DPU component",
                             COMPONENT_CATEGORY_PROCESSOR)

  /* Element Library Params */
  static std::vector<SST::ElementInfoParam> getComponentParams() {
    auto params = DRRAResource::getBaseParams();
    return params;
  }
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  /* Element Library Ports */
  static std::vector<SST::ElementInfoPort> getComponentPorts() {
    auto ports = DRRAResource::getBasePorts();
    return ports;
  }
  SST_ELI_DOCUMENT_PORTS(getComponentPorts())

  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()

  DPU(SST::ComponentId_t id, SST::Params &params);
  ~DPU() {};

  // SST lifecycle methods
  virtual void init(unsigned int phase) override;
  virtual void setup() override;
  virtual void complete(unsigned int phase) override;
  virtual void finish() override;

  bool clockTick(SST::Cycle_t currentCycle) override;
  void handleEventWithSlotID(SST::Event *event, uint32_t slot_id);

private:
  // buffers
  std::map<uint32_t, std::vector<uint8_t>> data_buffers;
  std::vector<uint8_t> accumulate_register;

  // Supported opcodes
  void decodeInstr(uint32_t instr) override;
  enum OpCode { REP, REPX, FSM, DPU_OP };
  void handleRep(uint32_t instr);
  void handleRepx(uint32_t instr);
  void handleFSM(uint32_t instr);
  void handleDPU(uint32_t instr);
  void handleOperation(std::string name,
                       std::function<uint64_t(uint64_t, uint64_t)> operation);

  // DPU modes
  enum DPU_MODE {
    IDLE,
    ADD,
    SUM_ACC,
    ADD_CONST,
    SUBT,
    SUBT_ABS,
    MODE_6,
    MULT,
    MULT_ADD,
    MULT_CONST,
    MAC,
    LD_IR,
    AXPY,
    MAX_MIN_ACC,
    MAX_MIN_CONST,
    MODE_15,
    MAX_MIN,
    SHIFT_L,
    SHIFT_R,
    SIGM,
    TANHYP,
    EXPON,
    LK_RELU,
    RELU,
    DIV,
    ACC_SOFTMAX,
    DIV_SOFTMAX,
    LD_ACC,
    SCALE_DW,
    SCALE_UP,
    MAC_INTER,
    MODE_31
  };

  // Map of DSU modes to handlers
  std::map<DPU_MODE, std::function<void()>> dsuHandlers = {
      {IDLE, [this] { out.output("IDLE\n"); }},
      {ADD,
       [this] {
         handleOperation("ADD", [](uint64_t a, uint64_t b) { return a + b; });
       }},
      {ADD_CONST,
       [this] {
         handleOperation("ADD_CONST",
                         [](uint64_t a, uint64_t b) { return a + b; });
       }},
      {SUBT,
       [this] {
         handleOperation("SUBT", [](uint64_t a, uint64_t b) { return a - b; });
       }},
      {SUBT_ABS,
       [this] {
         handleOperation("SUBT_ABS",
                         [](uint64_t a, uint64_t b) { return a - b; });
       }},
      {MULT,
       [this] {
         handleOperation("MULT", [](uint64_t a, uint64_t b) { return a * b; });
       }},
      {MULT_CONST,
       [this] {
         handleOperation("MULT_CONST",
                         [](uint64_t a, uint64_t b) { return a * b; });
       }},
      {LD_IR,
       [this] {
         handleOperation("LD_IR", [](uint64_t a, uint64_t b) { return b; });
       }},
  };

  std::function<void()> getDSUHandler(DPU_MODE mode) {
    if (dsuHandlers.find(mode) == dsuHandlers.end())
      out.fatal(CALL_INFO, -1, "DSU mode %d not implemented\n", mode);

    return dsuHandlers[mode];
  }

  // Events handlers list
  std::vector<std::function<void()>> eventsHandlers;
  int32_t lastRepLevel = -1;
  uint32_t current_event_number = 0;

  // FSMs
  static const uint32_t num_fsms = 4;
  uint32_t current_fsm = 0;
  std::function<void()> fsmHandlers[num_fsms];
};

#endif // _DPU_H