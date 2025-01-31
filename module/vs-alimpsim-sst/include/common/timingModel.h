#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

class TimingExpression;
class TimingEvent;
class TransitionOperator;
class RepetitionOperator;
class TimingState;

class TimingExpression {
public:
  virtual ~TimingExpression() {}
  virtual uint64_t scheduleEvents(TimingState &state,
                                  uint64_t startCycle) const {
    return 0;
  };
  virtual std::string toString() const = 0;
  virtual uint64_t lastEventId() const { return 0; }
  virtual std::string lastEventName() const { return ""; }
};

class TimingState {
private:
  std::shared_ptr<TimingExpression> expression;
  uint64_t eventCounter;
  uint64_t lastScheduledCycle;

public:
  TimingState() : eventCounter(0), lastScheduledCycle(0) {}

  TimingState(std::shared_ptr<TimingExpression> expression)
      : expression(expression), eventCounter(expression->lastEventId() + 1),
        lastScheduledCycle(0) {
    addEventName(expression->lastEventName());
  }

  uint64_t currentCycle = 0;
  std::map<uint64_t, std::set<std::shared_ptr<const TimingEvent>>>
      scheduledEvents;
  std::vector<std::string> eventNames;

  uint64_t getLastScheduledCycle() const { return lastScheduledCycle; }
  void updateLastScheduledCycle(uint64_t cycle) {
    lastScheduledCycle = std::max(lastScheduledCycle, cycle);
  }

  static TimingState createFromEvent(const std::string &name) {
    uint64_t eventCounter = 0;
    auto event = std::make_shared<TimingEvent>(name, eventCounter++);
    return TimingState(std::static_pointer_cast<TimingExpression>(event));
  }

  TimingState &addEvent(const std::string &name) {
    auto event = std::make_shared<TimingEvent>(name, eventCounter++);
    addEventName(name);
    expression = std::static_pointer_cast<TimingExpression>(event);
    return *this;
  }

  TimingState &addTransition(uint64_t delay, const std::string &nextEventName) {
    if (!expression) {
      throw std::runtime_error("Cannot add transition without an event");
    }
    auto nextEvent =
        std::make_shared<TimingEvent>(nextEventName, eventCounter++);
    auto transition =
        std::make_shared<TransitionOperator>(delay, expression, nextEvent);
    expression = std::static_pointer_cast<TimingExpression>(transition);
    return *this;
  }

  TimingState &addRepetition(uint64_t iterations, uint64_t step) {
    if (!expression) {
      throw std::runtime_error("Cannot add repetition without an event");
    }
    auto repetition =
        std::make_shared<RepetitionOperator>(iterations, step, expression);
    expression = std::static_pointer_cast<TimingExpression>(repetition);
    return *this;
  }

  TimingState &build() {
    if (!expression) {
      throw std::runtime_error("Cannot build timing state without events");
    }
    auto expression = this->expression;
    updateLastScheduledCycle(
        expression->scheduleEvents(*this, lastScheduledCycle));
    return *this;
  }

  std::shared_ptr<TimingExpression> getExpression() const { return expression; }

  void scheduleEvent(std::shared_ptr<const TimingEvent> event, uint64_t cycle) {
    scheduledEvents[cycle].insert(event);
  }

  std::set<std::shared_ptr<const TimingEvent>>
  getEventsForCycle(uint64_t cycle) {
    auto eventsIterator = scheduledEvents.find(cycle);
    if (eventsIterator != scheduledEvents.end()) {
      return eventsIterator->second; // Return the set of events
    }
    return std::set<std::shared_ptr<const TimingEvent>>(); // Return an empty
                                                           // set
  }

  bool findEventByName(const std::string &name) {
    return std::find(eventNames.begin(), eventNames.end(), name) !=
           eventNames.end();
  }

  void addEventName(const std::string &name) {
    if (findEventByName(name)) {
      throw std::runtime_error("Event with name " + name + " already exists");
    } else {
      eventNames.push_back(name);
    }
  }

  std::string toString() const { return expression->toString(); }
};

class TimingEvent : public TimingExpression,
                    public std::enable_shared_from_this<TimingEvent> {
private:
  std::string name;
  uint64_t eventNumber;

public:
  TimingEvent(const std::string &name, uint64_t eventNumber)
      : name(name), eventNumber(eventNumber) {}

  uint64_t scheduleEvents(TimingState &state,
                          uint64_t startCycle) const override {
    state.scheduleEvent(shared_from_this(), startCycle);
    state.updateLastScheduledCycle(startCycle);
    return startCycle;
  }

  std::string toString() const override {
    return "e" + std::to_string(eventNumber);
  }

  std::string getName() const { return name; }
  uint64_t getEventNumber() const { return eventNumber; }

  uint64_t lastEventId() const override { return eventNumber; }
  std::string lastEventName() const override { return name; }
};

class TransitionOperator : public TimingExpression {
private:
  uint64_t delay;
  std::shared_ptr<TimingExpression> from;
  std::shared_ptr<TimingExpression> to;

public:
  TransitionOperator(uint64_t delay, std::shared_ptr<TimingExpression> from,
                     std::shared_ptr<TimingExpression> to)
      : delay(delay), from(from), to(to) {}

  uint64_t scheduleEvents(TimingState &state,
                          uint64_t startCycle) const override {
    uint64_t fromCycle = from->scheduleEvents(state, startCycle);
    uint64_t toCycle = to->scheduleEvents(state, fromCycle + delay);
    return toCycle;
  }

  std::string toString() const override {
    return "T<" + std::to_string(delay) + ">(" + from->toString() + "," +
           to->toString() + ")";
  }

  uint64_t lastEventId() const override { return to->lastEventId(); }
  std::string lastEventName() const override { return to->lastEventName(); }
};

class RepetitionOperator : public TimingExpression {
private:
  uint64_t iterations;
  uint64_t step;
  std::shared_ptr<TimingExpression> expression;

public:
  RepetitionOperator(uint64_t iterations, uint64_t step,
                     std::shared_ptr<TimingExpression> expression)
      : iterations(iterations), step(step), expression(expression) {}

  uint64_t scheduleEvents(TimingState &state,
                          uint64_t startCycle) const override {
    uint64_t lastCycle = startCycle;
    for (uint64_t i = 0; i < iterations; i++) {
      lastCycle = expression->scheduleEvents(state, lastCycle);
      if (i < iterations - 1) {
        lastCycle += step;
      }
    }
    return lastCycle;
  }

  std::string toString() const override {
    return "R<" + std::to_string(iterations) + "," + std::to_string(step) +
           ">(" + expression->toString() + ")";
  }

  uint64_t lastEventId() const override { return expression->lastEventId(); }
  std::string lastEventName() const override {
    return expression->lastEventName();
  }
};