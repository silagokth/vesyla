#ifndef _DATAEVENT_H
#define _DATAEVENT_H

#include <sst/core/event.h>

using namespace SST;

class DataEvent : public Event {
public:
  DataEvent() {}
  ~DataEvent() {}

  // Data members
  std::vector<uint8_t> payload; // Payload data
  size_t size;                  // Size of the payload (in bits)

  enum PortType { WriteNarrow, ReadNarrow, WriteWide, ReadWide };
  PortType portType; // Port type

  DataEvent *clone() override { return new DataEvent(*this); }

  void serialize_order(SST::Core::Serialization::serializer &ser) override {
    Event::serialize_order(ser);
  }

  ImplementSerializable(DataEvent);
};

#endif // _DATAEVENT_H