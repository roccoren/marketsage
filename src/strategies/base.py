import logging

class Strategy:
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Strategy instance created")

    def analyze(self, data):
        """
        Analyze the given data and return trading signals.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            List of signal dictionaries with at least 'Type' and 'Price' keys
        """
        raise NotImplementedError("Subclasses must implement analyze")
