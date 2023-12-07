import verboselogs


class ResourcePool:
    def __init__(self):
        self.logger = verboselogs.VerboseLogger("vesim")
        self.resource_pool = {}

    def add(self, name_, r_):
        self.resource_pool[name_] = r_

    def remove(self, name_):
        if name_ in self.resource_pool:
            del self.resource_pool[name_]
        else:
            self.logger.critical("Resource Remove Error: " + name_)
            exit()

    def get(self, name_):
        if name_ in self.resource_pool:
            return self.resource_pool[name_]
        else:
            self.logger.critical("Resource Get Error: " + name_)
            exit()

    def exist(self, name_):
        if name_ in self.resource_pool:
            return True
        else:
            return False

    def set(self, name_, value_):
        if name_ in self.resource_pool:
            self.resource_pool[name_] = value_
        else:
            self.logger.critical("Resource Set Error: " + name_)
            exit()

    def dump(self):
        for x in self.resource_pool:
            print(x, ":")
            print(self.resource_pool[x])
