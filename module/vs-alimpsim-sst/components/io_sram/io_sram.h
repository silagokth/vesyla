#ifndef _IOSRAM_H
#define _IOSRAM_H

#include "drra.h"

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/sst_types.h>
#include <sst/core/timeConverter.h>

#include "sst/elements/memHierarchy/membackend/backing.h"

class ScratchBackendConvertor;

class IOSRAM : public DRRAResource {
public:
  /* Element Library Info */
  SST_ELI_REGISTER_COMPONENT(IOSRAM, "drra", "IOSRAM",
                             SST_ELI_ELEMENT_VERSION(1, 0, 0),
                             "IOSRAM component", COMPONENT_CATEGORY_MEMORY)

  /* Element Library Params */
  static std::vector<SST::ElementInfoParam> getComponentParams() {
    auto params = DRRAResource::getBaseParams();
    params.push_back({"access_time", "Time to access the IO buffer", "0ns"});
    params.push_back(
        {"backing", "Type of backing store (malloc, mmap)", "malloc"});
    params.push_back(
        {"backing_size_unit", "Size of the backing store", "1MiB"});
    params.push_back({"memory_file", "Memory file for mmap backing", ""});
    return params;
  }
  SST_ELI_DOCUMENT_PARAMS(getComponentParams())

  /* Element Library Ports */
  static std::vector<SST::ElementInfoPort> getComponentPorts() {
    auto ports = DRRAResource::getBasePorts();
    ports.push_back({"io_port", "Link to input or output buffer"});
    return ports;
  }
  SST_ELI_DOCUMENT_PORTS(getComponentPorts())

  SST_ELI_DOCUMENT_STATISTICS()
  SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()

  /* Constructor */
  IOSRAM(SST::ComponentId_t id, SST::Params &params);

  /* Destructor */
  ~IOSRAM() {
    if (backend)
      delete backend;
  };

  // SST lifecycle methods
  virtual void init(unsigned int phase) override;
  virtual void setup() override;
  virtual void complete(unsigned int phase) override;
  virtual void finish() override;

  bool clockTick(SST::Cycle_t currentCycle) override;

private:
  std::string access_time;

  SST::MemHierarchy::Backend::Backing *backend = nullptr;
  ScratchBackendConvertor *backendConvertor = nullptr;

  // Map ports to links
  enum PortMap {
    SRAMReadFromIO = 0,
    SRAMWriteToIO = 1,
    IOWriteToSRAM = 2,
    IOReadFromSRAM = 3,
    WriteBulk = 6,
    ReadBulk = 7
  };

  SST::Link *io_link = nullptr;
  SST::Link *self_link = nullptr;
  int64_t sram_read_from_io_address_buffer = -1;
  int64_t sram_read_from_io_initial_addr = -1;
  int64_t sram_write_to_io_address_buffer = -1;
  int64_t sram_write_to_io_initial_addr = -1;
  std::vector<uint8_t> from_io_data_buffer;
  std::vector<uint8_t> to_io_data_buffer;
  int64_t io_write_to_sram_address_buffer = -1;
  int64_t io_write_to_sram_initial_addr = -1;
  int64_t io_read_from_sram_address_buffer = -1;
  int64_t io_read_from_sram_initial_addr = -1;

  // Bulk read/write
  int64_t read_bulk_address_buffer = -1;
  int64_t write_bulk_address_buffer = -1;
  int64_t read_bulk_initial_addr = -1;
  int64_t write_bulk_initial_addr = -1;

  // Supported opcodes
  void decodeInstr(uint32_t instr) override;
  enum OpCode { REP = 0, REPX = 1, DSU = 6 };
  void handleRep(uint32_t instr);
  void handleRepx(uint32_t instr);
  void handleDSU(uint32_t instr);

  void readFromIO();
  void writeToIO();
  void writeToSRAM();
  void readFromSRAM();
  void writeBulk();
  void readBulk();

  int32_t agu_initial_addr = -1;
  uint32_t current_event_number = 0;
};

#endif // _IOSRAM_H