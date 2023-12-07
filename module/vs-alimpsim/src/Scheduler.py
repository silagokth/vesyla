import verboselogs


class Scheduler:
    def __init__(self, event_pool_, resource_pool_, handler_pool_, step_):
        self.clk = 0
        self.event_pool = event_pool_
        self.resource_pool = resource_pool_
        self.handler_pool = handler_pool_
        self.logger = verboselogs.VerboseLogger('vesim')
        self.step = step_

    def get_clk(self):
        return self.clk

    def exec_event(self, e_):
        status = self.handler_pool.get(e_[1])(
            self.clk, self.event_pool, self.resource_pool, self.handler_pool, e_[2])
        if not status:
            self.logger.critical("Event {} failed!".format(e_[1]))
            exit(-1)

    def run(self):
        self.logger.info("---- |Cycle {: <3}| ----".format(self.clk))
        while (True):
            event_list = self.event_pool.get_list(self.clk)
            # sort event_list based on event[3]: the priority
            event_list.sort(key=lambda x: self.event_pool.get(x)[
                            3], reverse=True)
            if (event_list):
                k = event_list[0]
                self.exec_event(self.event_pool.get(k))
                self.event_pool.remove(k)
            else:
                if (not self.event_pool.check(self.clk)):
                    self.logger.critical("At least one event is missing!")
                    exit()
                self.clk = self.clk+1
                if (self.event_pool.is_finished()):
                    break
                self.logger.info("---- |Cycle {: <3}| ----".format(self.clk))

            if (self.step):
                input()
