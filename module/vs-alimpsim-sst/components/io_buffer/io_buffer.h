#ifndef _IOBUFFER_H
#define _IOBUFFER_H

#include "drra.h"

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/sst_types.h>
#include <sst/core/timeConverter.h>

#include "sst/elements/memHierarchy/membackend/backing.h"

class ScratchBackendConvertor;

class IOBuffer : public DRRAComponent {
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(IOBuffer,   // Class name
                             "drra",     // Name of library
                             "IOBuffer", // Lookup name for component
                             SST_ELI_ELEMENT_VERSION(1, 0,
                                                     0), // Component version
                             "IOBuffer component",       // Description
                             COMPONENT_CATEGORY_MEMORY   // Category
  )

  // Add component-specific parameters
  static std::vector<SST::ElementInfoParam> getComponentParams() {
    auto params = DRRAComponent::getBaseParams();
    params.push_back({"access_time", "Time to access the IO buffer", "0ns"});
    params.push_back(
        {"backing", "Type of backing store (malloc, mmap)", "malloc"});
    params.push_back(
        {"backing_size_unit", "Size of the backing store", "1MiB"});
    params.push_back({"memory_file", "Memory file for mmap backing", ""});
    return params;
  }
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  SST_ELI_DOCUMENT_PORTS(
      {"col_port%(portnum)d",
       "Link(s) to DRRA columns (should be connected to slot 1 of top or "
       "bottom cells). Connect col_port0, col_port1, etc."})

  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()
  /* Constructor */
  IOBuffer(SST::ComponentId_t id, SST::Params &params);

  /* Destructor */
  ~IOBuffer() {
    if (backend)
      delete backend;
  };

  // SST lifecycle methods
  virtual void init(unsigned int phase) override;
  virtual void setup() override;
  virtual void complete(unsigned int phase) override;
  virtual void finish() override;

  // SST clock handler
  bool clockTick(SST::Cycle_t currentCycle) override;

  // SST event handler
  void handleEvent(SST::Event *event) override;

private:
  // SST::Output out;
  // std::string clock;
  // uint64_t printFrequency;
  // uint64_t io_data_width;
  std::string access_time;

  SST::MemHierarchy::Backend::Backing *backend = nullptr;
  ScratchBackendConvertor *backendConvertor = nullptr;

  std::vector<SST::Link *> columnLinks;
};

#endif // _IOBUFFER_H