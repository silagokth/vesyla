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

  enum PortType { ReadNarrow, ReadWide, WriteNarrow, WriteWide };
  PortType portType; // Port type

  DataEvent(PortType portType) : portType(portType) {
    switch (portType) {
    case WriteNarrow:
      setPriority(MEMEVENTPRIORITY + 1);
      break;
    case ReadNarrow:
      setPriority(MEMEVENTPRIORITY);
      break;
    case WriteWide:
      setPriority(MEMEVENTPRIORITY + 1);
      break;
    case ReadWide:
      setPriority(MEMEVENTPRIORITY);
      break;
    default:
      break;
    }
  }

  DataEvent *clone() override { return new DataEvent(*this); }

  void serialize_order(SST::Core::Serialization::serializer &ser) override {
    Event::serialize_order(ser);
  }

  ImplementSerializable(DataEvent);
};

#endif // _DATAEVENT_H