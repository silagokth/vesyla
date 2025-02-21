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
  void handleEvent(SST::Event *event) override;
  void handleEventWithSlotID(SST::Event *event, uint32_t slot_id);

private:
  // Links
  SST::Link *data_links[2];

  // buffers
  uint64_t data_buffers[2];

  // Supported opcodes
  void decodeInstr(uint32_t instr);
  enum OpCode { REP, REPX, FSM, DPU_OP };
  void handleRep(uint32_t instr);
  void handleRepx(uint32_t instr);
  void handleFSM(uint32_t instr);
  void handleDPU(uint32_t instr);

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
    MAC,
    MULT_CONST,
    ACCUMULATE,
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

  // DSU handlers
  void handleIdle(void);
  void handleAdd(void);
  void handleSumAcc(void);
  void handleSubt(void);
  void handleMode6(void);
  void handleMult(void);
  void handleMAC(void);
  void handleAccumulate(void);
  void handleLD_IR(void);
  void handleAXPY(void);
  void handleMaxMinAcc(void);
  void handleMode15(void);
  void handleMaxMin(void);
  void handleShiftL(void);
  void handleShiftR(void);
  void handleSigm(void);
  void handleTanhyp(void);
  void handleExpon(void);
  void handleLkRelu(void);
  void handleRelu(void);
  void handleDiv(void);
  void handleAccSoftmax(void);
  void handleDivSoftmax(void);
  void handleLdAcc(void);
  void handleScaleDw(void);
  void handleScaleUp(void);
  void handleMacInter(void);
  void handleMode31(void);

  // Map of DSU modes to handlers
  std::map<DPU_MODE, std::function<void()>> dsuHandlers = {
      {IDLE, std::bind(&DPU::handleIdle, this)},
      {ADD, std::bind(&DPU::handleAdd, this)},
      {SUM_ACC, std::bind(&DPU::handleSumAcc, this)},
      {ADD_CONST, std::bind(&DPU::handleAdd, this)},
      {SUBT, std::bind(&DPU::handleSubt, this)},
      {SUBT_ABS, std::bind(&DPU::handleSubt, this)},
      {MODE_6, std::bind(&DPU::handleMode6, this)},
      {MULT, std::bind(&DPU::handleMult, this)},
      {MAC, std::bind(&DPU::handleMAC, this)},
      {MULT_CONST, std::bind(&DPU::handleMult, this)},
      {ACCUMULATE, std::bind(&DPU::handleAccumulate, this)},
      {LD_IR, std::bind(&DPU::handleLD_IR, this)},
      {AXPY, std::bind(&DPU::handleAXPY, this)},
      {MAX_MIN_ACC, std::bind(&DPU::handleMaxMinAcc, this)},
      {MAX_MIN_CONST, std::bind(&DPU::handleMaxMin, this)},
      {MODE_15, std::bind(&DPU::handleMode15, this)},
      {MAX_MIN, std::bind(&DPU::handleMaxMin, this)},
      {SHIFT_L, std::bind(&DPU::handleShiftL, this)},
      {SHIFT_R, std::bind(&DPU::handleShiftR, this)},
      {SIGM, std::bind(&DPU::handleSigm, this)},
      {TANHYP, std::bind(&DPU::handleTanhyp, this)},
      {EXPON, std::bind(&DPU::handleExpon, this)},
      {LK_RELU, std::bind(&DPU::handleLkRelu, this)},
      {RELU, std::bind(&DPU::handleRelu, this)},
      {DIV, std::bind(&DPU::handleDiv, this)},
      {ACC_SOFTMAX, std::bind(&DPU::handleAccSoftmax, this)},
      {DIV_SOFTMAX, std::bind(&DPU::handleDivSoftmax, this)},
      {LD_ACC, std::bind(&DPU::handleLdAcc, this)},
      {SCALE_DW, std::bind(&DPU::handleScaleDw, this)},
      {SCALE_UP, std::bind(&DPU::handleScaleUp, this)},
      {MAC_INTER, std::bind(&DPU::handleMacInter, this)},
      {MODE_31, std::bind(&DPU::handleMode31, this)}};

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