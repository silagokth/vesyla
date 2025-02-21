#ifndef _MEMEVENTS_H
#define _MEMEVENTS_H

#include <sst/core/event.h>

using namespace SST;

class IOEvent : public Event {};

class IOReadRequest : public IOEvent {
public:
  IOReadRequest() {}
  ~IOReadRequest() {}

  // Data members
  uint32_t address;
  uint32_t size;
  uint32_t column_id;

  IOReadRequest *clone() override { return new IOReadRequest(*this); }

  void serialize_order(SST::Core::Serialization::serializer &ser) override {
    Event::serialize_order(ser);
  }

  ImplementSerializable(IOReadRequest);
};

class IOReadResponse : public IOEvent {
public:
  IOReadResponse() {}
  ~IOReadResponse() {}

  // Data members
  uint32_t address;
  std::vector<uint8_t> data;

  IOReadResponse *clone() override { return new IOReadResponse(*this); }

  void serialize_order(SST::Core::Serialization::serializer &ser) override {
    Event::serialize_order(ser);
  }

  void set_data(std::vector<uint8_t> data) { this->data = data; }

  ImplementSerializable(IOReadResponse);
};

class IOWriteRequest : public IOEvent {
public:
  IOWriteRequest() {}
  ~IOWriteRequest() {}

  // Data members
  uint32_t address;
  std::vector<uint8_t> data;

  IOWriteRequest *clone() override { return new IOWriteRequest(*this); }

  void serialize_order(SST::Core::Serialization::serializer &ser) override {
    Event::serialize_order(ser);
  }

  ImplementSerializable(IOWriteRequest);

  void set_data(std::vector<uint8_t> data) { this->data = data; }
};

class IOWriteResponse : public IOEvent {
public:
  IOWriteResponse() {}
  ~IOWriteResponse() {}

  // Data members
  uint32_t data;

  IOWriteResponse *clone() override { return new IOWriteResponse(*this); }

  void serialize_order(SST::Core::Serialization::serializer &ser) override {
    Event::serialize_order(ser);
  }

  ImplementSerializable(IOWriteResponse);
};

#endif // _MEMEVENTS_H