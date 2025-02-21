#include <functional>
#include <map>
#include <memory>
#include <set>
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
                                  uint64_t startCycle) const;
  virtual std::string toString() const = 0;
  virtual uint64_t lastEventId() const;
  virtual std::string lastEventName() const;
};

class TimingState {
private:
  std::shared_ptr<TimingExpression> expression;
  uint64_t eventCounter;
  uint64_t lastScheduledCycle;

public:
  TimingState();
  TimingState(std::shared_ptr<TimingExpression> expression);

  uint64_t currentCycle = 0;
  std::map<uint64_t, std::set<std::shared_ptr<const TimingEvent>>>
      scheduledEvents;
  std::vector<std::string> eventNames;

  uint64_t getLastScheduledCycle() const;
  void updateLastScheduledCycle(uint64_t cycle);
  static TimingState createFromEvent(const std::string &name);
  TimingState &addEvent(const std::string &name, std::function<void()> handler);
  TimingState &addTransition(uint64_t delay, const std::string &nextEventName,
                             std::function<void()> handler);
  TimingState &addRepetition(uint64_t iterations, uint64_t step);
  TimingState &build();
  std::shared_ptr<TimingExpression> getExpression() const;
  void scheduleEvent(std::shared_ptr<const TimingEvent> event, uint64_t cycle);
  std::set<std::shared_ptr<const TimingEvent>>
  getEventsForCycle(uint64_t cycle);
  bool findEventByName(const std::string &name);
  void addEventName(const std::string &name);
  std::string toString() const;
};

class TimingEvent : public TimingExpression,
                    public std::enable_shared_from_this<TimingEvent> {
private:
  std::string name;
  uint64_t eventNumber;
  std::function<void()> handler;

public:
  TimingEvent(const std::string &name, uint64_t eventNumber);
  uint64_t scheduleEvents(TimingState &state,
                          uint64_t startCycle) const override;
  std::string toString() const override;
  std::string getName() const;
  uint64_t getEventNumber() const;
  uint64_t lastEventId() const override;
  std::string lastEventName() const override;
  void execute() const;
  void setHandler(std::function<void()> handler);
};

class TransitionOperator : public TimingExpression {
private:
  uint64_t delay;
  std::shared_ptr<TimingExpression> from;
  std::shared_ptr<TimingExpression> to;

public:
  TransitionOperator(uint64_t delay, std::shared_ptr<TimingExpression> from,
                     std::shared_ptr<TimingExpression> to);
  uint64_t scheduleEvents(TimingState &state,
                          uint64_t startCycle) const override;
  std::string toString() const override;
  uint64_t lastEventId() const override;
  std::string lastEventName() const override;
};

class RepetitionOperator : public TimingExpression {
private:
  uint64_t iterations;
  uint64_t step;
  std::shared_ptr<TimingExpression> expression;

public:
  RepetitionOperator(uint64_t iterations, uint64_t step,
                     std::shared_ptr<TimingExpression> expression);
  uint64_t scheduleEvents(TimingState &state,
                          uint64_t startCycle) const override;
  std::string toString() const override;
  uint64_t lastEventId() const override;
  std::string lastEventName() const override;
};