import logging
import verboselogs


class HandlerPool:
    def __init__(self):
        self.handler_pool = {}
        self.logger = verboselogs.VerboseLogger('vesim')

    def add(self, name_, r_):
        self.handler_pool[name_] = r_

    def remove(self, name_):
        logger = logging.getLogger()
        if name_ in self.handler_pool:
            del self.handler_pool[name_]
        else:
            logger.fatal("Remove handler error: "+name_)

    def get(self, name_):
        logger = logging.getLogger()
        if name_ in self.handler_pool:
            return self.handler_pool[name_]
        else:
            logger.fatal("Get handler error: "+name_)

    def set(self, name_, value_):
        logger = logging.getLogger()
        if name_ in self.handler_pool:
            self.handler_pool[name_] = value_
        else:
            logger.fatal("Set handler error: "+name_)
