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

class IOBuffer : public Component {
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
    std::vector<SST::ElementInfoParam> params;
    params.push_back({"clock", "Clock frequency", "100MHz"});
    params.push_back(
        {"printFrequency", "Frequency to print tick messages", "1000"});
    params.push_back({"io_data_width", "Width of the IO data", "256"});
    params.push_back({"io_depth", "Depth of the IO buffer", "65536"});
    params.push_back({"word_bitwidth", "Width of the word", "16"});
    params.push_back({"access_time", "Time to access the IO buffer", "0ns"});
    params.push_back(
        {"backing", "Type of backing store (malloc, mmap)", "malloc"});
    params.push_back(
        {"backing_size_unit", "Size of the backing store", "1MiB"});
    params.push_back({"memory_file", "Memory file for mmap backing", ""});
    params.push_back({"read_only", "Read-only mode", "false"});
    params.push_back({"num_columns", "Number of columns", "1"});
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
  bool clockTick(SST::Cycle_t currentCycle);

  // SST event handler
  void handleEvent(SST::Event *event);
  void handleEventFromColumn(SST::Event *event, uint32_t column_id);

private:
  DRRAOutput out;
  std::string clock;
  uint64_t printFrequency;
  uint64_t io_data_width, io_depth;
  uint64_t word_bitwidth;
  std::string access_time;

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

  uint32_t num_columns;

  SST::MemHierarchy::Backend::Backing *backend = nullptr;
  ScratchBackendConvertor *backendConvertor = nullptr;
  bool read_only;

  int64_t read_address_buffer = -1;
  int64_t write_address_buffer = -1;
  std::vector<uint8_t> read_data_buffer, write_data_buffer;

  std::vector<SST::Link *> column_links;
};

#endif // _IOBUFFER_H