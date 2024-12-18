#ifndef _IOBUFFER_H
#define _IOBUFFER_H

#include <map>
#include <queue>

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/sst_types.h>
#include <sst/core/timeConverter.h>

#include "sst/elements/memHierarchy/memEvent.h"
#include "sst/elements/memHierarchy/memLinkBase.h"
#include "sst/elements/memHierarchy/membackend/backing.h"
#include "sst/elements/memHierarchy/util.h"

class ScratchBackendConvertor;

class IOBuffer : public SST::Component
{
public:
    /* Element Library Info */
    SST_ELI_REGISTER_COMPONENT(
        IOBuffer,                         // Class name
        "drra",                           // Name of library
        "IOBuffer",                       // Lookup name for component
        SST_ELI_ELEMENT_VERSION(1, 0, 0), // Component version
        "IOBuffer component",             // Description
        COMPONENT_CATEGORY_MEMORY         // Category
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "100MHz"},
        {"io_data_width", "Width of the IO data", "256"},
        {"access_time", "Time to access the IO buffer", "0ns"})

    SST_ELI_DOCUMENT_PORTS(
        {"slot_port%(portnum)d", "Link(s) to resources in slots. Connect slot_port0, slot_port1, etc."})

    SST_ELI_DOCUMENT_STATISTICS()
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS()
    /* Constructor */
    IOBuffer(SST::ComponentId_t id, SST::Params &params);

    /* Destructor */
    ~IOBuffer()
    {
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
    void handleMemoryEvent(SST::MemHierarchy::MemEvent *event);

private:
    SST::Output out;
    std::string clock;
    uint64_t io_data_width;
    std::string access_time;

    SST::MemHierarchy::Backend::Backing *backend = nullptr;
    ScratchBackendConvertor *backendConvertor = nullptr;

    std::vector<SST::Link *> slotLinks;
};

#endif // _IOBUFFER_H