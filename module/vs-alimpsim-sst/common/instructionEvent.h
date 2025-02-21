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

    bool isResourceInstruction()
    {
        return getMSBit() == 1;
    }

    int32_t getSlot()
    {
        if (!isResourceInstruction())
        {
            return -1; // Control instruction
        }

        // Return the 4-bits from bit 27 to 24 (slot number)
        return (instruction >> 24) & 0xF;
    };

    ImplementSerializable(InstrEvent);

private:
    uint32_t getMSBit()
    {
        return (instruction >> 31) & 0x1;
    }
};

#endif // _INSTREVENT_H