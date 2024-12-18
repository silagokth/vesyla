#ifndef _ACTEVENT_H
#define _ACTEVENT_H

#include <sst/core/event.h>

using namespace SST;

class ActEvent : public Event
{
public:
    ActEvent() {}
    ~ActEvent() {}

    // Data members
    uint32_t port;

    ActEvent *clone() override
    {
        return new ActEvent(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override
    {
        Event::serialize_order(ser);
    }

    ImplementSerializable(ActEvent);
};

#endif // _ACTEVENT_H