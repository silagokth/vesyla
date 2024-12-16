#ifndef _VECADD_H
#define _VECADD_H

#include <sst/core/component.h>
#include <sst/core/params.h>

using namespace std;
using namespace SST;

class VecAdd : public Component
{
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(
      VecAdd,                           // Class name
      "vec_add",                        // Name of library
      "VecAdd",                         // Lookup name for component
      SST_ELI_ELEMENT_VERSION(1, 0, 0), // Component version
      "VecAdd DPU component",           // Description
      COMPONENT_CATEGORY_PROCESSOR      // Category
  )

  SST_ELI_DOCUMENT_PARAMS(
      {"test", "Test parameter", "1"}, // Example parameter
  )

  SST_ELI_DOCUMENT_PORTS(
      {"controller_port", "Link to the controller"})
  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()

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
  bool clockTick(Cycle_t currentCycle);

  // SST event handler
  void handleEvent(Event *event);

private:
  Output out;
  Cycle_t printFrequency;

  Link *controllerLink;
};

#endif // _VECADD_H
