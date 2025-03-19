"""
Microsoft Qlib and RDAgent integration for stock signal prediction.
"""

from .predictor import QlibRDPredictor
from .utils import prepare_qlib_data

__all__ = ["QlibRDPredictor", "prepare_qlib_data"] 