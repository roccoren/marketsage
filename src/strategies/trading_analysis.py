from typing import List, Dict, Any
import pandas as pd
from .base import Strategy
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SignalAnalysis:
    def __init__(self, signals: List[Dict[str, Any]]):
        self.signals = pd.DataFrame(signals)
        # Parse timestamps flexibly
        def clean_ts(ts):
            # Remove microseconds and Z suffix
            ts = ts.split('.')[0].rstrip('Z')
            # Split date and time
            date_part, time_part = ts.split(' ')
            # Replace hyphens with colons in time part
            time_part = time_part.replace('-', ':')
            return f"{date_part} {time_part}"
        self.signals['Timestamp'] = pd.to_datetime(self.signals['Timestamp'].apply(clean_ts), format='%Y-%m-%d %H:%M:%S')
        
        # Filter for last 7 trading days
        last_7_days = datetime.now() - timedelta(days=7)
        self.signals = self.signals[self.signals['Timestamp'] >= last_7_days]
        if len(self.signals) == 0:
            logger.warning("No signals found in the last 7 trading days")

    def analyze_by_strategy(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group and analyze signals by strategy type"""
        strategy_signals = {}
        
        for strategy in self.signals['Strategy'].unique():
            strategy_df = self.signals[self.signals['Strategy'] == strategy]
            signals_summary = []
            
            for _, signal in strategy_df.iterrows():
                signals_summary.append({
                    'symbol': signal['Symbol'],
                    'signal_type': signal.get('Signal', 'Unknown'),
                    'confidence': signal.get('Confidence', 0),
                    'timestamp': signal['Timestamp'],
                    'additional_info': {k: v for k, v in signal.items() 
                                     if k not in ['Symbol', 'Strategy', 'Signal', 'Confidence', 'Timestamp', 'Exchange']}
                })
            
            strategy_signals[strategy] = signals_summary
            
        return strategy_signals

class OpportunityAnalysis:
    def __init__(self, signals: List[Dict[str, Any]]):
        self.signals = pd.DataFrame(signals)
        self.signals['Timestamp'] = pd.to_datetime(self.signals['Timestamp'])

    def find_high_conviction_opportunities(self) -> List[Dict[str, Any]]:
        """Identify and rank symbols with multiple aligned signals"""
        opportunities = []
        
        # Group signals by symbol
        symbol_groups = self.signals.groupby('Symbol')
        
        for symbol, group in symbol_groups:
            signal_count = len(group)
            unique_strategies = group['Strategy'].nunique()
            avg_confidence = group['Confidence'].mean()
            
            if unique_strategies >= 2:  # Require signals from at least 2 different strategies
                opportunities.append({
                    'symbol': symbol,
                    'signal_count': signal_count,
                    'strategy_count': unique_strategies,
                    'confidence_score': avg_confidence,
                    'signals': group.to_dict('records'),
                    'rank_score': (signal_count * avg_confidence * unique_strategies) / 100
                })
        
        # Sort by rank score in descending order
        opportunities.sort(key=lambda x: x['rank_score'], reverse=True)
        return opportunities

class TradingAnalysis:
    def __init__(self, signals: List[Dict[str, Any]], market_data: Dict[str, pd.DataFrame] = None):
        self.signals = signals
        self.market_data = market_data or {}
        self.signal_analysis = SignalAnalysis(signals)
        self.opportunity_analysis = OpportunityAnalysis(signals)

    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive trading analysis report"""
        report = {
            "signal_summary": self._generate_signal_summary(),
            "high_conviction_opportunities": self._generate_opportunity_summary(),
            "recommended_actions": self._generate_action_recommendations(),
            "risk_assessment": self._generate_risk_assessment(),
            "execution_strategy": self._generate_execution_strategy()
        }
        return report

    def _generate_signal_summary(self) -> Dict[str, Any]:
        """Generate strategy-wise signal summary"""
        strategy_signals = self.signal_analysis.analyze_by_strategy()
        
        summary = {}
        for strategy, signals in strategy_signals.items():
            summary[strategy] = {
                "total_signals": len(signals),
                "signals_by_type": {},
                "average_confidence": sum(s['confidence'] for s in signals) / len(signals) if signals else 0,
                "signals": sorted(signals, key=lambda x: x['timestamp'], reverse=True)  # Include all signals from last 7 days
            }
            
            # Count signals by type
            for signal in signals:
                signal_type = signal['signal_type']
                if signal_type not in summary[strategy]["signals_by_type"]:
                    summary[strategy]["signals_by_type"][signal_type] = 0
                summary[strategy]["signals_by_type"][signal_type] += 1
                
        return summary

    def _generate_opportunity_summary(self) -> List[Dict[str, Any]]:
        """Generate summary of high-conviction opportunities"""
        opportunities = self.opportunity_analysis.find_high_conviction_opportunities()
        
        for opportunity in opportunities:
            # Add entry/exit points if available in signals
            entry_points = []
            exit_points = []
            
            for signal in opportunity['signals']:
                if 'entry_price' in signal:
                    entry_points.append(signal['entry_price'])
                if 'exit_price' in signal:
                    exit_points.append(signal['exit_price'])
            
            if entry_points:
                opportunity['primary_entry'] = sum(entry_points) / len(entry_points)
            if exit_points:
                opportunity['primary_exit'] = sum(exit_points) / len(exit_points)
        
        return opportunities

    def _generate_action_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific trading recommendations"""
        opportunities = self.opportunity_analysis.find_high_conviction_opportunities()
        recommendations = []
        
        for opp in opportunities:
            signals = opp['signals']
            buy_signals = sum(1 for s in signals if s.get('Signal', '').upper() == 'BUY')
            sell_signals = sum(1 for s in signals if s.get('Signal', '').upper() == 'SELL')
            
            action = 'HOLD'
            if buy_signals > sell_signals and buy_signals >= 2:
                action = 'BUY'
            elif sell_signals > buy_signals and sell_signals >= 2:
                action = 'SELL'
                
            recommendations.append({
                'symbol': opp['symbol'],
                'signal_date': max(s['Timestamp'] for s in signals).strftime('%Y-%m-%d'),
                'action': action,
                'confidence_score': opp['confidence_score'],
                'entry_range': {
                    'low': min(s.get('entry_price', 0) for s in signals if s.get('entry_price', 0) > 0),
                    'high': max(s.get('entry_price', 0) for s in signals if s.get('entry_price', 0) > 0)
                } if any(s.get('entry_price', 0) > 0 for s in signals) else None,
                'trade_timing': {
                    'optimal_entry': 'Market open' if action == 'BUY' else 'Market close',
                    'valid_until': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                },
                'stop_loss': min(s.get('stop_loss', 0) for s in signals if s.get('stop_loss', 0) > 0) if any(s.get('stop_loss', 0) > 0 for s in signals) else None,
                'position_size': self._calculate_position_size(opp),
            })
            
        return recommendations

    def _calculate_position_size(self, opportunity: Dict[str, Any]) -> float:
        """Calculate suggested position size based on volatility and confidence"""
        base_size = 0.1  # 10% base position size
        confidence_modifier = opportunity['confidence_score'] / 100
        
        # Adjust based on volatility if market data is available
        if self.market_data and opportunity['symbol'] in self.market_data:
            data = self.market_data[opportunity['symbol']]
            returns = data['Close'].pct_change()
            volatility = returns.std()
            vol_modifier = 1 - (volatility * 10)  # Reduce size for high volatility
            return base_size * confidence_modifier * max(0.2, vol_modifier)
        
        return base_size * confidence_modifier

    def _generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate risk assessment summary"""
        risk_assessment = {
            "market_conditions": self._assess_market_conditions(),
            "volatility_measures": self._assess_volatility(),
            "correlation_analysis": self._analyze_correlations(),
            "volume_analysis": self._analyze_volume_profiles()
        }
        return risk_assessment

    def _assess_market_conditions(self) -> Dict[str, Any]:
        """Assess current market conditions"""
        # Simplified market condition assessment based on signal patterns
        recent_signals_df = pd.DataFrame(self.signals)
        if 'Timestamp' not in recent_signals_df.columns or len(recent_signals_df) == 0:
            return {"market_bias": "Neutral", "signal_strength": "Weak", "recent_signal_count": 0}
            
        recent_signals_df = recent_signals_df.sort_values('Timestamp', ascending=False).head(100)
        
        buy_ratio = len(recent_signals_df[recent_signals_df.get('Signal', pd.Series(['UNKNOWN'] * len(recent_signals_df))) == 'BUY']) / len(recent_signals_df)
        
        return {
            "market_bias": "Bullish" if buy_ratio > 0.6 else "Bearish" if buy_ratio < 0.4 else "Neutral",
            "signal_strength": "Strong" if abs(buy_ratio - 0.5) > 0.3 else "Moderate" if abs(buy_ratio - 0.5) > 0.1 else "Weak",
            "recent_signal_count": len(recent_signals_df)
        }

    def _assess_volatility(self) -> Dict[str, Any]:
        """Assess volatility measures"""
        volatility_measures = {}
        
        if self.market_data:
            for symbol, data in self.market_data.items():
                returns = data['Close'].pct_change()
                volatility_measures[symbol] = {
                    "daily_volatility": returns.std(),
                    "avg_true_range": self._calculate_atr(data)
                }
        
        return volatility_measures

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr

    def _analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between symbols"""
        correlations = {}
        
        if self.market_data and len(self.market_data) > 1:
            returns_dict = {}
            for symbol, data in self.market_data.items():
                returns_dict[symbol] = data['Close'].pct_change()
            
            returns_df = pd.DataFrame(returns_dict)
            correlation_matrix = returns_df.corr()
            
            # Calculate average correlation for each symbol
            for symbol in correlation_matrix.index:
                correlations[symbol] = correlation_matrix[symbol].mean()
        
        return correlations

    def _analyze_volume_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Analyze volume profiles"""
        volume_profiles = {}
        
        if self.market_data:
            for symbol, data in self.market_data.items():
                recent_volume = data['Volume'].tail(20)  # Last 20 periods
                avg_volume = recent_volume.mean()
                
                volume_profiles[symbol] = {
                    "average_volume": avg_volume,
                    "volume_trend": "Increasing" if recent_volume.iloc[-1] > avg_volume else "Decreasing",
                    "volume_stability": "Stable" if recent_volume.std() / avg_volume < 0.5 else "Volatile"
                }
        
        return volume_profiles

    def _generate_execution_strategy(self) -> Dict[str, Any]:
        """Generate execution strategy recommendations"""
        opportunities = self.opportunity_analysis.find_high_conviction_opportunities()
        
        execution_strategy = {
            "order_types": self._recommend_order_types(opportunities),
            "timeframes": self._recommend_timeframes(opportunities),
            "risk_parameters": self._recommend_risk_parameters(opportunities),
            "position_management": self._recommend_position_management(opportunities)
        }
        
        return execution_strategy

    def _recommend_order_types(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend appropriate order types based on opportunities"""
        recommendations = {}
        
        for opp in opportunities:
            volatility = (
                self.market_data[opp['symbol']]['Close'].pct_change().std()
                if self.market_data and opp['symbol'] in self.market_data
                else None
            )
            
            if volatility is None:
                recommendations[opp['symbol']] = {
                    "primary": "LIMIT",
                    "backup": "STOP_LIMIT",
                    "reasoning": "Default conservative approach without volatility data"
                }
            elif volatility > 0.02:  # High volatility
                recommendations[opp['symbol']] = {
                    "primary": "STOP_LIMIT",
                    "backup": "LIMIT",
                    "reasoning": "High volatility protection"
                }
            else:
                recommendations[opp['symbol']] = {
                    "primary": "LIMIT",
                    "backup": "MARKET",
                    "reasoning": "Low volatility allows more aggressive entry"
                }
                
        return recommendations

    def _recommend_timeframes(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend trading timeframes based on opportunities"""
        recommendations = {}
        
        for opp in opportunities:
            signals = opp['signals']
            signal_timeframes = [s.get('timeframe', 'daily') for s in signals]
            
            # Count occurrences of each timeframe
            timeframe_counts = {}
            for tf in signal_timeframes:
                timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
                
            # Get most common timeframe
            primary_timeframe = max(timeframe_counts.items(), key=lambda x: x[1])[0]
            
            recommendations[opp['symbol']] = {
                "primary": primary_timeframe,
                "alternatives": list(set(signal_timeframes) - {primary_timeframe}),
                "reasoning": f"Based on {len(signals)} signals across {len(set(signal_timeframes))} timeframes"
            }
            
        return recommendations

    def _recommend_risk_parameters(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend risk management parameters"""
        recommendations = {}
        
        for opp in opportunities:
            symbol = opp['symbol']
            volatility = (
                self.market_data[symbol]['Close'].pct_change().std() * 100
                if self.market_data and symbol in self.market_data
                else None
            )
            
            base_stop_loss = 2.0  # Base stop loss percentage
            if volatility is not None:
                # Adjust stop loss based on volatility
                stop_loss = base_stop_loss * (1 + volatility)
            else:
                stop_loss = base_stop_loss
                
            recommendations[symbol] = {
                "stop_loss_percent": round(stop_loss, 2),
                "take_profit_ratio": 2,  # Risk:Reward ratio
                "max_loss_per_trade": "1%",
                "trailing_stop": "Optional" if volatility and volatility < 2 else "Recommended"
            }
            
        return recommendations

    def _recommend_position_management(self, opportunities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Recommend position management guidelines"""
        base_guidelines = [
            "Scale in positions when multiple signals confirm",
            "Use trailing stops on profitable positions",
            "Consider partial profit taking at key resistance levels",
            "Adjust position size based on volatility"
        ]
        
        volatile_guidelines = [
            "Reduce position size by 50%",
            "Tighten stops during high volatility periods",
            "Use options for downside protection"
        ]
        
        low_vol_guidelines = [
            "Consider pyramiding on continued strength",
            "Wider stops to avoid noise",
            "Full position sizing acceptable"
        ]
        
        guidelines = {
            "general": base_guidelines,
            "high_volatility": volatile_guidelines,
            "low_volatility": low_vol_guidelines
        }
        
        return guidelines