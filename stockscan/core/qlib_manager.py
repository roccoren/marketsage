"""
QlibManager - Handles Qlib initialization and data preparation
"""
import os
from typing import List, Dict, Any, Optional, Union
import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.filter import NameDFilter, ExpressionDFilter

class QlibManager:
    """
    Manager class for Qlib initialization and data operations
    """
    
    def __init__(self, provider_uri: str = '~/.qlib/qlib_data/cn_data', 
                 region: str = 'cn', 
                 cache_level: int = 1):
        """
        Initialize the QlibManager
        
        Parameters
        ----------
        provider_uri : str
            The URI of the data provider
        region : str
            The region of the data
        cache_level : int
            The level of cache (0, 1, or 2)
        """
        self.provider_uri = provider_uri
        self.region = region
        self.cache_level = cache_level
        self.initialized = False
        
    def init(self):
        """
        Initialize Qlib with the specified parameters
        """
        if not self.initialized:
            qlib.init(provider_uri=self.provider_uri, 
                     region=self.region, 
                     expression_cache={
                        "class": "DiskExpressionCache",
                        "kwargs": {}
                     },
                     dataset_cache={
                        "class": "SimpleDatasetCache",
                        "kwargs": {}
                     },
                     auto_mount=True)
            self.initialized = True
            
    def filter_instruments_by_name(self, name_rule_re: str) -> List[str]:
        """
        Filter instruments by name rule
        
        Parameters
        ----------
        name_rule_re : str
            Name rule regular expression
            
        Returns
        -------
        List[str]
            List of filtered instrument names
        """
        self._ensure_initialized()
        name_filter = NameDFilter(name_rule_re=name_rule_re)
        return name_filter.get_instruments()
    
    def filter_instruments_by_expression(self, expression: str) -> List[str]:
        """
        Filter instruments by expression
        
        Parameters
        ----------
        expression : str
            Expression for filtering
            
        Returns
        -------
        List[str]
            List of filtered instrument names
        """
        self._ensure_initialized()
        expr_filter = ExpressionDFilter(expression)
        return expr_filter.get_instruments()
    
    def create_dataset_handler(self, 
                            instruments: Union[List[str], str], 
                            start_time: str, 
                            end_time: str, 
                            fields: List[str],
                            freq: str = 'day') -> DataHandlerLP:
        """
        Create a DataHandler for a specific dataset
        
        Parameters
        ----------
        instruments : Union[List[str], str]
            Instruments to include in the dataset
        start_time : str
            Start time of the dataset
        end_time : str
            End time of the dataset
        fields : List[str]
            Fields to include in the dataset
        freq : str
            Frequency of the dataset
            
        Returns
        -------
        DataHandlerLP
            The data handler for the dataset
        """
        self._ensure_initialized()
        return DataHandlerLP(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader={
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": fields,
                        "freq": freq,
                    }
                }
            }
        )
    
    def prepare_dataset(self, 
                       handler: DataHandlerLP, 
                       segments: Dict[str, tuple]) -> DatasetH:
        """
        Prepare a dataset with specified segments
        
        Parameters
        ----------
        handler : DataHandlerLP
            DataHandler for the dataset
        segments : Dict[str, tuple]
            Segments for the dataset, e.g., {'train': ('2010-01-01', '2018-12-31'), 
                                           'valid': ('2019-01-01', '2019-12-31'),
                                           'test': ('2020-01-01', '2021-12-31')}
            
        Returns
        -------
        DatasetH
            The prepared dataset
        """
        self._ensure_initialized()
        return DatasetH(handler, segments)
    
    def _ensure_initialized(self):
        """
        Ensure Qlib is initialized
        """
        if not self.initialized:
            self.init() 