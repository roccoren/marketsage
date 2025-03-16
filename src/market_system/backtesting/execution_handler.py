"""
Execution handler for simulating trade execution in backtesting.
Handles slippage, commission, and realistic order execution.
"""

from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)

class ExecutionHandler:
    """
    Simulates trade execution with realistic assumptions.
    Handles slippage, commission, and execution delays.
    """
    
    def __init__(self, 
                 slippage_model: str = "percentage",
                 slippage_params: Optional[Dict] = None,
                 commission_model: str = "percentage",
                 commission_params: Optional[Dict] = None):
        """
        Initialize the execution handler.
        
        Args:
            slippage_model: Type of slippage model ("fixed", "percentage", "volatility")
            slippage_params: Parameters for the slippage model
            commission_model: Type of commission model ("fixed", "percentage")
            commission_params: Parameters for the commission model
        """
        self.slippage_model = slippage_model
        # More realistic slippage parameters
        self.slippage_params = slippage_params or {"percentage": 0.001}  # 0.1% slippage by default
        self.commission_model = commission_model
        # More realistic commission parameters
        self.commission_params = commission_params or {"percentage": 0.001, "min_commission": Decimal("1.0")}  # 0.1% commission with $1 minimum
        
        logger.info(f"ExecutionHandler initialized with slippage model: {slippage_model}, commission model: {commission_model}")
        
    def execute_order(self, 
                      symbol: str, 
                      action: str, 
                      price: Decimal, 
                      quantity: Decimal,
                      timestamp: datetime,
                      market_data: Optional[Dict] = None,
                      stop_loss: Optional[Decimal] = None,
                      take_profit: Optional[Decimal] = None) -> Dict:
        """
        Simulate order execution with slippage and commission.
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            price: Theoretical execution price
            quantity: Order quantity
            timestamp: Execution timestamp
            market_data: Optional market data for more realistic simulation
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            Dict with execution details
        """
        # Calculate slippage
        slippage = self.calculate_slippage(symbol, action, price, quantity, timestamp, market_data)
        
        # Apply slippage to price
        if action == "BUY":
            executed_price = price + slippage
        else:  # SELL
            executed_price = price - slippage
            
        # Calculate commission
        commission = self.calculate_commission(symbol, executed_price, quantity)
        
        # Calculate total cost
        total_cost = executed_price * quantity
        
        # Calculate net cost including commission
        if action == "BUY":
            net_cost = total_cost + commission
        else:  # SELL
            net_cost = total_cost - commission
            
        # Create execution report
        execution = {
            "symbol": symbol,
            "action": action,
            "requested_price": price,
            "executed_price": executed_price,
            "quantity": quantity,
            "timestamp": timestamp,
            "slippage": slippage,
            "commission": commission,
            "total_cost": total_cost,
            "net_cost": net_cost,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        
        logger.debug(f"Executed order: {symbol} {action} {quantity} @ {executed_price} (slippage: {slippage}, commission: {commission})")
        
        return execution
        
    def calculate_slippage(self, 
                           symbol: str, 
                           action: str, 
                           price: Decimal,
                           quantity: Decimal,
                           timestamp: datetime,
                           market_data: Optional[Dict] = None) -> Decimal:
        """
        Calculate slippage for an order.
        
        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            price: Theoretical execution price
            quantity: Order quantity
            timestamp: Execution timestamp
            market_data: Optional market data for volatility-based slippage
            
        Returns:
            Slippage amount
        """
        if self.slippage_model == "fixed":
            # Fixed pip slippage
            fixed_pips = Decimal(str(self.slippage_params.get("fixed_pips", 1)))
            pip_value = Decimal("0.0001")  # Adjust based on asset type
            return fixed_pips * pip_value
            
        elif self.slippage_model == "percentage":
            # Percentage of price slippage
            percentage = Decimal(str(self.slippage_params.get("percentage", 0.001)))
            return price * percentage
            
        elif self.slippage_model == "volatility":
            # Volatility-based slippage
            if market_data and symbol in market_data and "atr_14" in market_data[symbol]:
                # Use ATR as a measure of volatility
                atr = Decimal(str(market_data[symbol]["atr_14"]))
                atr_multiplier = Decimal(str(self.slippage_params.get("atr_multiplier", 0.1)))
                return atr * atr_multiplier
            else:
                # Fall back to percentage if ATR not available
                percentage = Decimal(str(self.slippage_params.get("percentage", 0.001)))
                return price * percentage
                
        elif self.slippage_model == "random":
            # Random slippage within a range
            min_percentage = Decimal(str(self.slippage_params.get("min_percentage", 0.0)))
            max_percentage = Decimal(str(self.slippage_params.get("max_percentage", 0.002)))
            random_percentage = Decimal(str(random.uniform(float(min_percentage), float(max_percentage))))
            return price * random_percentage
            
        else:
            logger.warning(f"Unknown slippage model: {self.slippage_model}, using zero slippage")
            return Decimal("0.0")
        
    def calculate_commission(self, 
                             symbol: str, 
                             price: Decimal,
                             quantity: Decimal) -> Decimal:
        """
        Calculate commission for an order.
        
        Args:
            symbol: Trading symbol
            price: Execution price
            quantity: Order quantity
            
        Returns:
            Commission amount
        """
        if self.commission_model == "fixed":
            # Fixed commission per trade
            return Decimal(str(self.commission_params.get("fixed_amount", 0.0)))
            
        elif self.commission_model == "percentage":
            # Percentage of trade value
            percentage = Decimal(str(self.commission_params.get("percentage", 0.001)))
            min_commission = Decimal(str(self.commission_params.get("min_commission", 1.0)))
            trade_value = price * quantity
            calculated_commission = trade_value * percentage
            
            # Apply minimum commission
            return max(calculated_commission, min_commission)
            
        elif self.commission_model == "per_share":
            # Fixed amount per share/contract
            per_share = Decimal(str(self.commission_params.get("per_share", 0.01)))
            return per_share * quantity
            
        elif self.commission_model == "tiered":
            # Tiered commission based on quantity
            trade_value = price * quantity
            
            if trade_value <= Decimal("1000"):
                return Decimal(str(self.commission_params.get("tier1", 5.0)))
            elif trade_value <= Decimal("10000"):
                return Decimal(str(self.commission_params.get("tier2", 10.0)))
            else:
                return Decimal(str(self.commission_params.get("tier3", 15.0)))
                
        else:
            logger.warning(f"Unknown commission model: {self.commission_model}, using zero commission")
            return Decimal("0.0") 