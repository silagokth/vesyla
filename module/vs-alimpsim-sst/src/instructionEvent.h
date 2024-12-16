#ifndef _INSTREVENT_H
#define _INSTREVENT_H

#include <sst/core/event.h>

using namespace SST;

class InstrEvent : public Event
{
public:
    InstrEvent() {}
    ~InstrEvent() {}

    // Data members
    uint32_t instruction;
    int32_t cycleToExecute = -1;

    InstrEvent *clone() override
    {
        return new InstrEvent(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override
    {
        Event::serialize_order(ser);
    }

    ImplementSerializable(InstrEvent);
};

#endif // _INSTREVENT_H