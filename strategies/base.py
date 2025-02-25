import logging

class Strategy:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.debug("Strategy instance created")
    def generate_signal(self, data):
        raise NotImplementedError("Subclasses must implement generate_signal")
