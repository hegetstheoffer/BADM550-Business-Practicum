import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use the list of the top 50 US tech companies by market cap
tickers = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-A', 'TSM', 'LLY', 'AVGO',
    'TSLA', 'WMT', 'JPM', 'UNH', 'XOM', 'V', 'ORCL', 'MA', 'HD', 'PG',
    'NVO', 'COST', 'JNJ', 'ABBV', 'ASML', 'BAC', 'NFLX', 'KO', 'MRK', 'CVX',
    'CRM', 'AMD', 'SAP', 'BABA', 'TMUS', 'ACN', 'PEP', 'AZN', 'NVS', 'TM',
    'TMO', 'LIN', 'MCD', 'ADBE', 'CSCO', 'IBM', 'WFC', 'GE', 'ABT', 'PDD'
]

# Function to fetch the current 3-month T-bill rate from Yahoo Finance
def get_tbill_rate():
    try:
        tbill = yf.Ticker("^IRX")
        tbill_rate = tbill.history(period='1d')['Close'].iloc[-1] / 100  # Convert to decimal
        print(f"Fetched 3-month T-bill rate: {tbill_rate * 100:.2f}%")
        return tbill_rate
    except Exception as e:
        print(f"Error fetching T-bill rate: {e}")
        return 0.05  # Default to 5% if fetching fails

# Get the current market caps and prices of these companies
def get_market_caps_and_prices(tickers):
    market_caps = {}
    current_prices = {}
    valid_tickers = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', None)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            if market_cap is not None and not np.isnan(current_price):
                market_caps[ticker] = market_cap
                current_prices[ticker] = current_price
                valid_tickers.append(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return market_caps, current_prices, valid_tickers

market_caps, current_prices, tickers = get_market_caps_and_prices(tickers)

# Get 5-year historical data
def get_historical_data(tickers):
    try:
        historical_data = yf.download(tickers, period='5y')['Adj Close']
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        historical_data = pd.DataFrame()
    return historical_data

historical_data = get_historical_data(tickers)

# Define market-cap weighted portfolio
def calculate_portfolio_weights(market_caps):
    total_market_cap = sum(market_caps.values())
    weights = {ticker: market_caps[ticker] / total_market_cap for ticker in market_caps}
    return weights

portfolio_weights = calculate_portfolio_weights(market_caps)

# Simulate portfolio performance
def backtest_portfolio(historical_data, portfolio_weights, initial_investment=1000000):
    returns = historical_data.pct_change().dropna()

    # Portfolio returns (weighted sum of individual stock returns)
    weighted_returns = returns.dot(pd.Series(portfolio_weights))

    # Portfolio value over time
    portfolio_value = (1 + weighted_returns).cumprod() * initial_investment

    return portfolio_value, weighted_returns

portfolio_value, portfolio_returns = backtest_portfolio(historical_data, portfolio_weights)

# Performance metrics adjusted for T-bill rate
def calculate_performance_metrics(portfolio_returns, tbill_rate):
    total_return = (portfolio_value[-1] / portfolio_value[0]) - 1

    # Converting 3-month T-bill rate to daily rate
    daily_tbill_rate = (1 + tbill_rate)**(1/90) - 1

    # Excess returns (portfolio returns minus daily risk-free rate)
    excess_returns = portfolio_returns - daily_tbill_rate

    # Sharpe Ratio: mean(excess return) / std(return) * sqrt(252)
    sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)

    # Max Drawdown: maximum peak-to-trough drop in portfolio value
    max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()

    return total_return, sharpe_ratio, max_drawdown

# Fetch the current T-bill rate from Yahoo Finance
tbill_rate = get_tbill_rate()

# Use the fetched T-bill rate for the Sharpe Ratio calculation
total_return, sharpe_ratio, max_drawdown = calculate_performance_metrics(portfolio_returns, tbill_rate)

# Output performance metrics
print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio (based on T-bill rate): {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plot portfolio value over time
plt.figure(figsize=(10,6))
plt.plot(portfolio_value, label="Portfolio Value")
plt.title("Portfolio Backtest Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()
