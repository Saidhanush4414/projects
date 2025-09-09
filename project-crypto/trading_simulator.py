"""
Trading Simulator for Whale Detection Strategy
Simulates trading based on whale detection signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class TradingSimulator:
    """Simulates trading based on whale detection signals"""
    
    def __init__(self):
        self.initial_balance = 10000.0
        self.position_size = 0.1  # 10% of balance per trade
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.04  # 4% take profit
        
    def run_simulation(self, portfolio_state, params=None):
        """Run trading simulation based on whale detection
        params: optional dict with keys { 'coin': str, 'amount': float }
        """
        try:
            coin = 'BTC'
            fixed_amount = None
            if isinstance(params, dict):
                coin = params.get('coin', coin) or 'BTC'
                fixed_amount = params.get('amount', None)
            # Get whale events
            from whale_detector import WhaleDetector
            whale_detector = WhaleDetector()
            whale_data = whale_detector.detect_events()
            
            # Ensure we have the expected data structure
            if 'events' not in whale_data:
                whale_data['events'] = []
            if 'detection_summary' not in whale_data:
                whale_data['detection_summary'] = {'total_events': 0}
            
            # Update portfolio based on whale events
            updated_portfolio = self._process_whale_events(portfolio_state, whale_data['events'], coin, fixed_amount)
            
            # Calculate performance metrics
            performance = self._calculate_performance(updated_portfolio)
            
            return {
                'portfolio_state': updated_portfolio,
                'performance': performance,
                'whale_events': whale_data['events'],
                'predictions': whale_data.get('predictions', {}),
                'trades': updated_portfolio['trades'][-10:]  # Last 10 trades
            }
            
        except Exception as e:
            print(f"Error in trading simulation: {e}")
            # Return safe fallback data
            return {
                'portfolio_state': portfolio_state,
                'performance': {
                    'total_return': 0,
                    'current_equity': portfolio_state.get('balance', 10000),
                    'total_trades': len(portfolio_state.get('trades', [])),
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'max_drawdown': 0,
                    'win_rate': 0
                },
                'whale_events': [],
                'predictions': {
                    'market_outlook': 'Error - Unable to analyze whale activity',
                    'price_prediction': 'No prediction available',
                    'volatility_forecast': 'Unable to forecast',
                    'trading_recommendation': 'HOLD - Error in analysis'
                },
                'trades': []
            }
    
    def _process_whale_events(self, portfolio_state, whale_events, coin_symbol='BTC', fixed_amount=None):
        """Process whale events and update portfolio"""
        try:
            portfolio = portfolio_state.copy()
            # Very rough spot prices by symbol for demo
            price_map = {
                'BTC': 45000.0,
                'ETH': 2500.0,
                'SOL': 150.0,
                'ADA': 0.5,
                'XRP': 0.6,
            }
            current_price = price_map.get((coin_symbol or 'BTC').upper(), 100.0)
            
            # Ensure portfolio has required fields
            if 'trades' not in portfolio:
                portfolio['trades'] = []
            if 'positions' not in portfolio:
                portfolio['positions'] = {}
            if 'equity_history' not in portfolio:
                portfolio['equity_history'] = []
            if 'realized_profit' not in portfolio:
                portfolio['realized_profit'] = 0.0
            
            for event in whale_events:
                trade_decision = self._make_trade_decision(event, portfolio, coin_symbol)
                
                if trade_decision['action'] == 'BUY' and portfolio['balance'] > 0:
                    # Execute buy order
                    trade_amount = (fixed_amount if isinstance(fixed_amount, (int, float)) and fixed_amount > 0 else portfolio['balance'] * self.position_size)
                    trade_amount = min(trade_amount, portfolio['balance'])
                    quantity = trade_amount / current_price
                    
                    trade = {
                        'timestamp': event.get('timestamp', datetime.now().isoformat()),
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': quantity,
                        'amount': trade_amount,
                        'symbol': coin_symbol.upper(),
                        'whale_event': event,
                        'reason': f"Whale {event.get('event_type', 'unknown')} detected with {event.get('confidence', 0):.2f} confidence"
                    }
                    
                    portfolio['trades'].append(trade)
                    portfolio['balance'] -= trade_amount
                    
                    sym = coin_symbol.upper()
                    if sym not in portfolio['positions']:
                        portfolio['positions'][sym] = 0
                    portfolio['positions'][sym] += quantity
                    
                elif trade_decision['action'] == 'SELL' and portfolio['positions'].get(coin_symbol.upper(), 0) > 0:
                    # Execute sell order (only if in profit)
                    sym = coin_symbol.upper()
                    coin_quantity = portfolio['positions'][sym]
                    if coin_quantity > 0:
                        # Check if we're in profit
                        avg_buy_price = self._get_average_buy_price(portfolio['trades'], sym)
                        if current_price > avg_buy_price:
                            sell_amount = coin_quantity * current_price
                            # compute realized P&L
                            realized_pnl = (current_price - avg_buy_price) * coin_quantity
                            
                            trade = {
                                'timestamp': event.get('timestamp', datetime.now().isoformat()),
                                'action': 'SELL',
                                'price': current_price,
                                'quantity': coin_quantity,
                                'amount': sell_amount,
                                'symbol': sym,
                                'whale_event': event,
                                'reason': f"Whale {event.get('event_type', 'unknown')} detected, selling at profit",
                                'pnl': realized_pnl
                            }
                            
                            portfolio['trades'].append(trade)
                            portfolio['balance'] += sell_amount
                            portfolio['positions'][sym] = 0
                            portfolio['realized_profit'] += realized_pnl
                
                # Update equity history
                current_equity = portfolio['balance'] + sum(
                    qty * (price_map.get(symbol, current_price))
                    for symbol, qty in portfolio['positions'].items()
                )
                portfolio['equity_history'].append({
                    'timestamp': event.get('timestamp', datetime.now().isoformat()),
                    'equity': current_equity,
                    'balance': portfolio['balance'],
                    'positions_value': current_equity - portfolio['balance']
                })
            
            return portfolio
            
        except Exception as e:
            print(f"Error processing whale events: {e}")
            return portfolio_state
    
    def _make_trade_decision(self, whale_event, portfolio, coin_symbol='BTC'):
        """Make trading decision based on whale event"""
        try:
            # Only trade if confidence is high enough
            confidence = whale_event.get('confidence', 0)
            if confidence < 0.6:
                return {'action': 'HOLD', 'reason': 'Low confidence signal'}
            
            event_type = whale_event.get('event_type', 'unknown')
            
            # If whale is buying and we have no position, consider buying
            sym = (coin_symbol or 'BTC').upper()
            if event_type == 'buying' and portfolio['positions'].get(sym, 0) == 0:
                return {'action': 'BUY', 'reason': 'Whale buying detected'}
            
            # If whale is selling and we have a position, consider selling (only if profitable)
            elif event_type == 'selling' and portfolio['positions'].get(sym, 0) > 0:
                return {'action': 'SELL', 'reason': 'Whale selling detected'}
            
            return {'action': 'HOLD', 'reason': 'No clear signal'}
            
        except Exception as e:
            print(f"Error making trade decision: {e}")
            return {'action': 'HOLD', 'reason': 'Error in analysis'}
    
    def _get_average_buy_price(self, trades, symbol_filter=None):
        """Calculate average buy price from trades (optionally for a specific symbol)"""
        buy_trades = [t for t in trades if t['action'] == 'BUY' and (symbol_filter is None or t.get('symbol') == symbol_filter)]
        if not buy_trades:
            return 0
        
        total_cost = sum(t['amount'] for t in buy_trades)
        total_quantity = sum(t['quantity'] for t in buy_trades)
        
        return total_cost / total_quantity if total_quantity > 0 else 0
    
    def _calculate_performance(self, portfolio):
        """Calculate portfolio performance metrics"""
        if not portfolio['equity_history']:
            return {
                'total_return': 0,
                'current_equity': portfolio['balance'],
                'total_trades': len(portfolio['trades']),
                'winning_trades': 0,
                'losing_trades': 0,
                'max_drawdown': 0
            }
        
        initial_equity = self.initial_balance
        current_equity = portfolio['equity_history'][-1]['equity']
        total_return = (current_equity - initial_equity) / initial_equity
        
        # Calculate win/loss ratio
        completed_trades = self._get_completed_trades(portfolio['trades'])
        winning_trades = sum(1 for trade in completed_trades if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in completed_trades if trade['pnl'] < 0)
        
        # Calculate max drawdown
        equity_values = [h['equity'] for h in portfolio['equity_history']]
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        return {
            'total_return': round(total_return * 100, 2),
            'current_equity': round(current_equity, 2),
            'total_trades': len(portfolio['trades']),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'max_drawdown': round(max_drawdown * 100, 2),
            'win_rate': round(winning_trades / len(completed_trades) * 100, 2) if completed_trades else 0
        }
    
    def _get_completed_trades(self, trades):
        """Get completed buy-sell trade pairs"""
        completed = []
        buy_trades = []
        
        for trade in trades:
            if trade['action'] == 'BUY':
                buy_trades.append(trade)
            elif trade['action'] == 'SELL' and buy_trades:
                buy_trade = buy_trades.pop(0)
                pnl = (trade['price'] - buy_trade['price']) * trade['quantity']
                completed.append({
                    'buy': buy_trade,
                    'sell': trade,
                    'pnl': pnl
                })
        
        return completed
    
    def _calculate_max_drawdown(self, equity_values):
        """Calculate maximum drawdown"""
        if not equity_values:
            return 0
        
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
