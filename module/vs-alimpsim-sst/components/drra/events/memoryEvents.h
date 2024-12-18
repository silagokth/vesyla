#ifndef _MEMEVENTS_H
#define _MEMEVENTS_H

#include <sst/core/event.h>

using namespace SST;

class ReadRequest : public Event
{
public:
    ReadRequest() {}
    ~ReadRequest() {}

    // Data members
    uint32_t address;

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

class ReadResponse : public Event
{
public:
    ReadResponse() {}
    ~ReadResponse() {}

    // Data members
    uint32_t data;

    ReadResponse *clone() override
    {
        return new ReadResponse(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer &ser) override
    {
        Event::serialize_order(ser);
    }

    ImplementSerializable(ReadResponse);
};

class WriteRequest : public Event
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

class WriteResponse : public Event
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