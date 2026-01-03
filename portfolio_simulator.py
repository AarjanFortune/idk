import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Note: Since we only have images of the CSV files, I'll create sample data structures
# In production, you would load: 
# predictions_df = pd.read_csv('model_predictions-gru.csv')
# stock_df = pd.read_csv('stock_with_sentiment_aggregated.csv')

# For demonstration, creating sample data based on the images shown
# You'll need to replace this with actual CSV file loading

def load_data(predictions_path, stock_path):
    """
    Load prediction and stock data
    """
    predictions_df = pd.read_csv(predictions_path)
    stock_df = pd.read_csv(stock_path)
    
    # Ensure date columns are datetime
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])
    
    # Parse the stock data date column (assuming first column is date)
    stock_df.iloc[:, 0] = pd.to_datetime(stock_df.iloc[:, 0])
    
    return predictions_df, stock_df


class PortfolioSimulator:
    def __init__(self, predictions_df, stock_df, initial_capital=10000, commission_rate=0.001):
        """
        Initialize portfolio simulator
        
        Parameters:
        - predictions_df: DataFrame with columns [date, actual, predicted, predicted_prob, prediction_date]
        - stock_df: DataFrame with stock data including open, high, low, close prices
        - initial_capital: Starting capital amount
        - commission_rate: Commission rate as decimal (e.g., 0.001 = 0.1% per trade)
        """
        self.predictions_df = predictions_df.copy()
        self.stock_df = stock_df.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.total_commission_paid = 0
        
        # Parse stock data columns
        self.stock_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume'] + list(self.stock_df.columns[6:])
        self.stock_df['date'] = pd.to_datetime(self.stock_df['date'])
        
        # Sort by date
        self.predictions_df = self.predictions_df.sort_values('date').reset_index(drop=True)
        self.stock_df = self.stock_df.sort_values('date').reset_index(drop=True)
        
        # Create date to price mapping for quick lookup
        self.price_map = self.stock_df.set_index('date')[['open', 'close']].to_dict('index')
        
        # Trading log
        self.trades = []
        self.portfolio_value = []
        
    def get_price(self, date, price_type='open'):
        """Get stock price for a specific date"""
        if date in self.price_map:
            return self.price_map[date][price_type]
        return None
    
    def get_next_trading_day(self, current_date, days_ahead):
        """
        Get the trading day that is 'days_ahead' trading days from current_date
        """
        available_dates = sorted(self.stock_df['date'].unique())
        
        if current_date not in available_dates:
            # Find the next available trading day
            future_dates = [d for d in available_dates if d > current_date]
            if not future_dates:
                return None
            current_date = future_dates[0]
        
        current_idx = available_dates.index(current_date)
        target_idx = current_idx + days_ahead
        
        if target_idx < len(available_dates):
            return available_dates[target_idx]
        return None
    
    def simulate(self):
        """
        Run the portfolio simulation based on model predictions
        
        Strategy:
        - On date D, model predicts if stock will go up in next 5 trading days
        - If prediction is UP (predicted=1), buy at open of D+1, sell at close of D+5
        - If prediction is DOWN (predicted=0), stay in cash
        - Only long positions allowed
        """
        position = None  # Current position: None or dict with {entry_date, entry_price, shares, exit_date}
        
        for idx, pred_row in self.predictions_df.iterrows():
            prediction_date = pred_row['date']
            predicted = pred_row['predicted']
            
            # Check if we're currently in a position
            if position is not None:
                # Check if we've reached the exit date
                if prediction_date >= position['exit_date']:
                    # Close the position
                    exit_price = self.get_price(position['exit_date'], 'close')
                    
                    if exit_price is not None:
                        gross_proceeds = position['shares'] * exit_price
                        exit_commission = gross_proceeds * self.commission_rate
                        net_proceeds = gross_proceeds - exit_commission
                        
                        total_commission = position['entry_commission'] + exit_commission
                        pnl = net_proceeds - position['investment']
                        pnl_pct = (pnl / position['investment']) * 100
                        
                        self.current_capital = net_proceeds
                        self.total_commission_paid += exit_commission
                        
                        self.trades.append({
                            'prediction_date': position['prediction_date'],
                            'entry_date': position['entry_date'],
                            'exit_date': position['exit_date'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'investment': position['investment'],
                            'gross_proceeds': gross_proceeds,
                            'entry_commission': position['entry_commission'],
                            'exit_commission': exit_commission,
                            'total_commission': total_commission,
                            'net_proceeds': net_proceeds,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'capital_after': self.current_capital
                        })
                    
                    position = None
            
            # If no position and model predicts UP, enter a position
            if position is None and predicted == 1:
                # Entry is next trading day (D+1)
                entry_date = self.get_next_trading_day(prediction_date, 1)
                
                if entry_date is None:
                    continue
                
                # Exit is 5 trading days after entry (D+1+5 = D+6 from prediction)
                exit_date = self.get_next_trading_day(entry_date, 4)  # 4 more days after entry = 5 days total
                
                if exit_date is None:
                    continue
                
                entry_price = self.get_price(entry_date, 'open')
                
                if entry_price is not None and self.current_capital > 0:
                    # Calculate shares accounting for commission on both entry and exit
                    # Total commission = commission_rate * investment (entry) + commission_rate * proceeds (exit)
                    # Since proceeds ≈ investment (approximately), we reserve 2 * commission_rate * investment
                    shares = self.current_capital / (entry_price * (1 + self.commission_rate))
                    investment = shares * entry_price
                    entry_commission = investment * self.commission_rate
                    
                    position = {
                        'prediction_date': prediction_date,
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'investment': investment,
                        'entry_commission': entry_commission
                    }
                    
                    self.current_capital = 0  # All capital invested
                    self.total_commission_paid += entry_commission
            
            # Track portfolio value
            current_value = self.current_capital
            if position is not None:
                current_price = self.get_price(prediction_date, 'close')
                if current_price is not None:
                    current_value += position['shares'] * current_price
            
            self.portfolio_value.append({
                'date': prediction_date,
                'value': current_value
            })
        
        # Close any remaining position at the last available date
        if position is not None:
            last_date = self.stock_df['date'].max()
            exit_price = self.get_price(last_date, 'close')
            
            if exit_price is not None:
                gross_proceeds = position['shares'] * exit_price
                exit_commission = gross_proceeds * self.commission_rate
                net_proceeds = gross_proceeds - exit_commission
                
                total_commission = position['entry_commission'] + exit_commission
                pnl = net_proceeds - position['investment']
                pnl_pct = (pnl / position['investment']) * 100
                
                self.current_capital = net_proceeds
                self.total_commission_paid += exit_commission
                
                self.trades.append({
                    'prediction_date': position['prediction_date'],
                    'entry_date': position['entry_date'],
                    'exit_date': last_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'investment': position['investment'],
                    'gross_proceeds': gross_proceeds,
                    'entry_commission': position['entry_commission'],
                    'exit_commission': exit_commission,
                    'total_commission': total_commission,
                    'net_proceeds': net_proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_after': self.current_capital
                })
    
    def get_performance_metrics(self):
        """Calculate and return performance metrics"""
        if not self.trades:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_gain': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'final_capital': self.current_capital,
                'total_commission_paid': self.total_commission_paid,
                'commission_rate': self.commission_rate * 100  # Convert to percentage
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_return = self.current_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_commission_paid': self.total_commission_paid,
            'commission_rate': self.commission_rate * 100,  # Convert to percentage
            'num_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0,
            'avg_gain': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'avg_gain_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'best_trade': trades_df['pnl'].max(),
            'worst_trade': trades_df['pnl'].min(),
            'best_trade_pct': trades_df['pnl_pct'].max(),
            'worst_trade_pct': trades_df['pnl_pct'].min(),
        }
        
        return metrics
    
    def get_trades_df(self):
        """Return trades as DataFrame"""
        return pd.DataFrame(self.trades)
    
    def get_portfolio_value_df(self):
        """Return portfolio value over time as DataFrame"""
        return pd.DataFrame(self.portfolio_value)
    
    def plot_results(self, save_path=None):
        """Create visualization of portfolio performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Simulation Results', fontsize=16, fontweight='bold')
        
        # Portfolio value over time
        portfolio_df = self.get_portfolio_value_df()
        if not portfolio_df.empty:
            axes[0, 0].plot(portfolio_df['date'], portfolio_df['value'], linewidth=2, color='#2E86AB')
            axes[0, 0].axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital', alpha=0.7)
            axes[0, 0].set_title('Portfolio Value Over Time', fontweight='bold')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Trade P&L distribution
        trades_df = self.get_trades_df()
        if not trades_df.empty:
            axes[0, 1].hist(trades_df['pnl_pct'], bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_title('Trade P&L Distribution (%)', fontweight='bold')
            axes[0, 1].set_xlabel('P&L (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative returns
        if not trades_df.empty:
            trades_df['cumulative_return'] = trades_df['pnl'].cumsum()
            axes[1, 0].plot(range(len(trades_df)), trades_df['cumulative_return'], linewidth=2, color='#F18F01', marker='o', markersize=4)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Cumulative Returns by Trade', fontweight='bold')
            axes[1, 0].set_xlabel('Trade Number')
            axes[1, 0].set_ylabel('Cumulative P&L ($)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics summary
        metrics = self.get_performance_metrics()
        metrics_text = f"""
        Initial Capital: ${metrics['initial_capital']:,.2f}
        Final Capital: ${metrics['final_capital']:,.2f}
        Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)
        
        Number of Trades: {metrics['num_trades']}
        Winning Trades: {metrics['winning_trades']}
        Losing Trades: {metrics['losing_trades']}
        Win Rate: {metrics['win_rate']:.2f}%
        
        Avg Gain: ${metrics['avg_gain']:,.2f} ({metrics['avg_gain_pct']:.2f}%)
        Avg Loss: ${metrics['avg_loss']:,.2f} ({metrics['avg_loss_pct']:.2f}%)
        
        Best Trade: ${metrics['best_trade']:,.2f} ({metrics['best_trade_pct']:.2f}%)
        Worst Trade: ${metrics['worst_trade']:,.2f} ({metrics['worst_trade_pct']:.2f}%)
        """
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Example usage
if __name__ == "__main__":
    # Load your actual CSV files
    # predictions_df, stock_df = load_data('model_predictions-gru.csv', 'stock_with_sentiment_aggregated.csv')
    
    # For demonstration, you would initialize like this:
    # simulator = PortfolioSimulator(predictions_df, stock_df, initial_capital=10000)
    # simulator.simulate()
    # metrics = simulator.get_performance_metrics()
    # trades_df = simulator.get_trades_df()
    # simulator.plot_results(save_path='portfolio_results.png')
    
    print("Portfolio Simulator Ready!")
    print("="*60)
    print("\nTo use this simulator with your CSV files:")
    print("1. Ensure your CSV files are accessible")
    print("2. Load them using: predictions_df, stock_df = load_data('predictions.csv', 'stock.csv')")
    print("3. Initialize: simulator = PortfolioSimulator(predictions_df, stock_df, initial_capital=10000)")
    print("4. Run simulation: simulator.simulate()")
    print("5. Get results: metrics = simulator.get_performance_metrics()")
    print("6. View trades: trades_df = simulator.get_trades_df()")
    print("7. Plot results: simulator.plot_results()")
