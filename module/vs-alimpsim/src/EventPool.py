class EventPool:

    def __init__(self):
        self.event_pool = {}
        self.event_counter = 0

    def post(self, e_):
        id = self.event_counter
        self.event_counter = self.event_counter+1
        self.event_pool[id] = e_

    def remove(self, id_):
        if id_ in self.event_pool:
            del self.event_pool[id_]
        else:
            print("Event Remove Error!")
            exit()

    def get(self, id_):
        if id_ in self.event_pool:
            return self.event_pool[id_]
        else:
            print("Event Get Error!")
            exit()

    def get_list(self, clk_):
        ret_list = []
        for k in self.event_pool:
            e = self.event_pool[k]
            if e[0] == clk_:
                ret_list.append(k)
        return ret_list

    def check(self, clk_):
        for k in self.event_pool:
            e = self.event_pool[k]
            if e[0] < clk_:
                return False
        return True

    def size(self):
        return len(self.event_pool)

    def dump_event_time(self):
        t = []
        for k in self.event_pool:
            e = self.event_pool[k]
            t.append(e[0])
        print(t)

    def is_finished(self):
        active_event_count = 0
        for k in self.event_pool:
            e = self.event_pool[k]
            if e[4]:
                # This event is important event
                active_event_count += 1
        if active_event_count == 0:
            return True
        else:
            return False
