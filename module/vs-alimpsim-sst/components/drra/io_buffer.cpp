#include "io_buffer.h"

#include <sst/core/componentExtension.h>
#include <sst/core/interfaces/stdMem.h>

#include "memoryEvents.h"

IOBuffer::IOBuffer(SST::ComponentId_t id, SST::Params &params) : SST::Component(id)
{
    out.init("IOBuffer[@p:@l]: ", 16, 0, SST::Output::STDOUT);
    clock = params.find<std::string>("clock", "100MHz");
    io_data_width = params.find<uint64_t>("io_data_width", 256);
    access_time = params.find<std::string>("access_time", "0ns");

    // Clock
    SST::TimeConverter *tc = registerClock(clock, new SST::Clock::Handler<IOBuffer>(this, &IOBuffer::clockTick));

    // Backing store
    bool found = false;
    std::string backingType = params.find<std::string>("backing", "malloc", found); /* Default to using an mmap backing store, fall back on malloc */
    if (!found)
    {
        bool oldBackVal = params.find<bool>("do-not-back", false, found);
        if (oldBackVal)
            backingType = "none";
    }

    // Backend
    std::string mallocSize = params.find<std::string>("backing_size_unit", "1MiB");
    SST::UnitAlgebra size(mallocSize);
    if (!size.hasUnits("B"))
    {
        out.fatal(CALL_INFO, -1, "Invalid memory size specified: %s\n", mallocSize.c_str());
    }
    size_t sizeBytes = size.getRoundedValue();

    // Create the backend
    if (backingType == "mmap")
    {
        std::string memoryFile = params.find<std::string>("memory_file", "");
        if (0 == memoryFile.compare(""))
        {
            memoryFile.clear();
        }
        try
        {
            backend = new SST::MemHierarchy::Backend::BackingMMAP(memoryFile, sizeBytes);
        }
        catch (int e)
        {
            if (e == 1)
            {
                out.fatal(CALL_INFO, -1, "Failed to open memory file: %s\n", memoryFile.c_str());
            }
            else
            {
                out.fatal(CALL_INFO, -1, "Failed to map memory file: %s\n", memoryFile.c_str());
            }
        }
    }
    else if (backingType == "malloc")
    {
        backend = new SST::MemHierarchy::Backend::BackingMalloc(sizeBytes);
    }
    out.output("IOBuffer (ID:%lu): Created backing store (type: %s)\n", id, backingType.c_str());

    // Slots ports
    std::string linkPrefix = "slot_port";
    std::string linkName = linkPrefix + "0";
    int portNum = 0;
    while (isPortConnected(linkName))
    {
        SST::Link *link = configureLink(linkName, access_time, new SST::Event::Handler<IOBuffer>(this, &IOBuffer::handleEvent));
        sst_assert(link, CALL_INFO, -1, "Failed to configure link %s\n", linkName.c_str());

        if (!link)
        {
            out.fatal(CALL_INFO, -1, "Failed to configure link %s\n", linkName.c_str());
        }
        slotLinks.push_back(link);

        // SST::Params memlink = params.get_scoped_params(linkName);
        // memlink.insert("port", linkName);
        // memLinks.push_back(loadAnonymousSubComponent<SST::MemHierarchy::MemLinkBase>("memHierarchy.MemLink", "cpulink", portNum, SST::ComponentInfo::SHARE_PORTS | SST::ComponentInfo::INSERT_STATS, memlink, tc));
        // out.output("HERE\n");
        // memLinks[portNum]->setRecvHandler(new SST::Event::Handler<IOBuffer>(this, &IOBuffer::handleEvent));

        // Next link
        portNum++;
        linkName = linkPrefix + std::to_string(portNum);
    }
}

void IOBuffer::init(unsigned int phase)
{
    out.output("IOBuffer initializing\n");
    // SST::Event *event = nullptr;
    // int count = 0;
    // for (int i = 0; i < slotLinks.size(); i++)
    // {
    // slotLinks[i]->sendUntimedData(new SST::MemHierarchy::MemEventInitCoherence(getName(), SST::MemHierarchy::Endpoint::Scratchpad, true, true, io_data_width / 8, true));
    // while (slotLinks[i]->recvUntimedData())
    // {
    // event = slotLinks[i]->recvUntimedData();
    // SST::MemHierarchy::MemEventInit *initEvent = dynamic_cast<SST::MemHierarchy::MemEventInit *>(event);
    // if (initEvent->getCmd() == SST::MemHierarchy::Command::NULLCMD)
    // {
    //     if (initEvent->getInitCmd() == SST::MemHierarchy::MemEventInit::InitCommand::Coherence)
    //     {
    //         SST::MemHierarchy::MemEventInitCoherence *initCoherence = dynamic_cast<SST::MemHierarchy::MemEventInitCoherence *>(initEvent);
    //         if (initCoherence->getType() == SST::MemHierarchy::Endpoint::Cache)
    //         {
    //             out.output("IOBuffer received cache init event\n");
    //             out.fatal(CALL_INFO, -1, "test");
    //         }
    //         else if (initCoherence->getType() == SST::MemHierarchy::Endpoint::Directory)
    //         {
    //             out.output("IOBuffer received directory init event\n");
    //             out.fatal(CALL_INFO, -1, "test");
    //         }
    //         else if (initCoherence->getType() == SST::MemHierarchy::Endpoint::Memory)
    //         {
    //             out.output("IOBuffer received memory init event\n");
    //             out.fatal(CALL_INFO, -1, "test");
    //         }
    //         else if (initCoherence->getType() == SST::MemHierarchy::Endpoint::Scratchpad)
    //         {
    //             out.output("IOBuffer received scratchpad init event\n");
    //             out.fatal(CALL_INFO, -1, "test");
    //         }
    //         else
    //         {
    //             out.fatal(CALL_INFO, -1, "Invalid init event (unknown type)\n");
    //         }
    //     }
    // }
    // else
    // {
    //     out.fatal(CALL_INFO, -1, "Invalid init event (no lower memory level)\n");
    // }
    // if (count == 100)
    // {
    //     out.fatal(CALL_INFO, -1, "test");
    // }
    // count++;
    // }
    // }
    out.verbose(CALL_INFO, 1, 0, "IOBuffer initialized\n");
}

void IOBuffer::setup()
{
    // out.verbose(CALL_INFO, 1, 0, "IOBuffer setup\n");
}

void IOBuffer::complete(unsigned int phase)
{
    // out.verbose(CALL_INFO, 1, 0, "IOBuffer completed\n");
}

void IOBuffer::finish()
{
    // out.verbose(CALL_INFO, 1, 0, "IOBuffer finishing\n");
}

bool IOBuffer::clockTick(SST::Cycle_t currentCycle)
{
    // out.verbose(CALL_INFO, 1, 0, "IOBuffer clock tick\n");
    return false;
}

void IOBuffer::handleEvent(SST::Event *event)
{
    ReadRequest *readReq = dynamic_cast<ReadRequest *>(event);
    if (readReq)
    {
        out.output("IOBuffer received read request\n");
    }

    WriteRequest *writeReq = dynamic_cast<WriteRequest *>(event);
    if (writeReq)
    {
        out.output("IOBuffer received write request\n");
    }
}

void IOBuffer::handleMemoryEvent(SST::MemHierarchy::MemEvent *event)
{
    if (event->getCmd() == SST::MemHierarchy::Command::GetSResp)
    {
        if (backend)
        {
            backend->set(event->getAddr(), event->getPayloadSize(), event->getPayload());
        }
    }

    out.output("IOBuffer received event\n");
    // Check if the event is a memory request
    SST::Interfaces::StandardMem::Read *readReq = dynamic_cast<SST::Interfaces::StandardMem::Read *>(event);
    if (readReq)
    {
        out.output("IOBuffer received read request\n");
    }

    SST::Interfaces::StandardMem::Write *writeReq = dynamic_cast<SST::Interfaces::StandardMem::Write *>(event);
    if (writeReq)
    {
        out.output("IOBuffer received write request\n");
    }
}