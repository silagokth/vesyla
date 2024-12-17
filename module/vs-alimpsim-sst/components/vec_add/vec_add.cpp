#include <sst/core/component.h>

#include "instructionEvent.h"
#include "vec_add.h"

VecAdd::VecAdd(ComponentId_t id, Params &params) : Component(id)
{
    out.init("VecAdd[@p:@l]: ", 16, 0, Output::STDOUT);
    printFrequency = params.find<uint64_t>("printFrequency", 1000);

    registerClock("100MHz", new SST::Clock::Handler<VecAdd>(this, &VecAdd::clockTick));
}

VecAdd::~VecAdd() {}

void VecAdd::init(unsigned int phase)
{
    controllerLink = configureLink("controller_port", "0ns", new Event::Handler<VecAdd>(this, &VecAdd::handleEvent));
    out.verbose(CALL_INFO, 1, 0, "VecAdd initialized\n");
}

void VecAdd::setup()
{
    // out.verbose(CALL_INFO, 1, 0, "VecAdd setup\n");
}

void VecAdd::complete(unsigned int phase)
{
    // out.verbose(CALL_INFO, 1, 0, "VecAdd completed\n");
}

void VecAdd::finish()
{
    // out.verbose(CALL_INFO, 1, 0, "VecAdd finishing\n");
}

bool VecAdd::clockTick(Cycle_t currentCycle)
{
    // if (currentCycle % printFrequency == 0)
    // {
    //     out.output("VecAdd tick at %" PRIu64 "\n", currentCycle);
    // }

    return false;
}

void VecAdd::handleEvent(Event *event)
{
    Event *newEvent = event;
    uint32_t instruction = 0;
    out.output("VecAdd received event\n");

    if (newEvent)
    {
        // Check if the event is an ActEvent
        ActEvent *actEvent = dynamic_cast<ActEvent *>(newEvent);
        if (actEvent)
        {
            out.output("VecAdd received ActEvent\n");
            active = true;
        }

        // Check if the event is an InstrEvent
        InstrEvent *instrEvent = dynamic_cast<InstrEvent *>(newEvent);
        if (instrEvent)
        {
            instrBuffer = instrEvent->instruction;
            out.output("VecAdd received InstrEvent: %08x\n", instrBuffer);
        }

        // Delete the event
        delete newEvent;
    }
}