#ifndef _MEMEVENTS_H
#define _MEMEVENTS_H

#include <sst/core/event.h>

using namespace SST;

class MemoryEvent : public Event
{
};

class ReadRequest : public MemoryEvent
{
public:
    ReadRequest() {}
    ~ReadRequest() {}

    // Data members
    uint32_t address;
    uint32_t size;
    uint32_t column_id;

    ReadRequest *clone() override
    {
        return new ReadRequest(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override
    {
        Event::serialize_order(ser);
    }

    ImplementSerializable(ReadRequest);
};

class ReadResponse : public MemoryEvent
{
public:
    ReadResponse() {}
    ~ReadResponse() {}

    // Data members
    uint32_t address;
    std::vector<uint8_t> data;

    ReadResponse *clone() override
    {
        return new ReadResponse(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override
    {
        Event::serialize_order(ser);
    }

    void set_data(std::vector<uint8_t> data)
    {
        this->data = data;
    }

    ImplementSerializable(ReadResponse);
};

class WriteRequest : public MemoryEvent
{
public:
    WriteRequest() {}
    ~WriteRequest() {}

    // Data members
    uint32_t address;
    std::vector<uint8_t> data;

    WriteRequest *clone() override
    {
        return new WriteRequest(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override
    {
        Event::serialize_order(ser);
    }

    ImplementSerializable(WriteRequest);

    void set_data(std::vector<uint8_t> data)
    {
        this->data = data;
    }
};

class WriteResponse : public MemoryEvent
{
public:
    WriteResponse() {}
    ~WriteResponse() {}

    // Data members
    uint32_t data;

    WriteResponse *clone() override
    {
        return new WriteResponse(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override
    {
        Event::serialize_order(ser);
    }

    ImplementSerializable(WriteResponse);
};

#endif // _MEMEVENTS_H