# Simple Backtest - Notebooks

Interactive Jupyter notebooks demonstrating the features and capabilities of the simple-backtest framework.

## 📚 Notebook Overview

### 1️⃣ [Basic Usage](01_basic_usage.ipynb)
**Start here if you're new to the framework!**

Learn the fundamentals:
- Loading data with yfinance
- Setting up commissions
- Running backtests with Moving Average strategy
- Comparing multiple strategies (Buy & Hold, DCA, Moving Average)
- Simple parameter optimization with Grid Search

**Prerequisites:** None
**Duration:** 15-20 minutes
**Difficulty:** Beginner

---

### 2️⃣ [Candlestick Pattern Strategies](02_candle_strategies.ipynb)
Create strategies based on candlestick patterns:
- Bullish/Bearish Engulfing
- Hammer and Shooting Star
- Doji patterns
- Morning and Evening Star
- Strategy comparison and analysis

**Prerequisites:** 01_basic_usage
**Duration:** 20-25 minutes
**Difficulty:** Intermediate

---

### 3️⃣ [Technical Analysis Strategies](03_ta_strategies.ipynb)
Build strategies using popular technical indicators:
- **Bollinger Bands** - Mean reversion
- **RSI** - Momentum and overbought/oversold
- **MACD** - Trend following
- **Stochastic Oscillator** - Momentum indicator
- **OBV** - Volume-based indicators

**Prerequisites:** 01_basic_usage
**Duration:** 25-30 minutes
**Difficulty:** Intermediate

---

### 4️⃣ [Machine Learning Strategies](04_ml_strategies.ipynb)
Apply machine learning to trading strategies:
- Feature engineering from OHLCV data
- Logistic Regression classifier
- Random Forest ensemble
- Gradient Boosting
- Walk-forward validation to prevent overfitting

**Prerequisites:** 01_basic_usage, 03_ta_strategies
**Duration:** 30-40 minutes
**Difficulty:** Advanced

---

### 5️⃣ [Commission Models](05_commission_usage.ipynb)
Master commission structures and their impact:
- Percentage commission (most common)
- Flat commission (fixed fee)
- Tiered commission (volume discounts)
- Creating custom commission models
- Commission impact analysis

**Prerequisites:** 01_basic_usage
**Duration:** 20-25 minutes
**Difficulty:** Intermediate

---

### 6️⃣ [Advanced Optimization](06_advanced_optimization.ipynb)
Advanced parameter optimization techniques:
- **Grid Search** - Exhaustive search
- **Random Search** - Efficient sampling
- **Walk-Forward Optimization** - Prevent overfitting
- Creating custom optimizers
- Best practices and pitfall avoidance

**Prerequisites:** 01_basic_usage
**Duration:** 30-40 minutes
**Difficulty:** Advanced

---

## 🚀 Getting Started

### Installation

```bash
# Install dependencies with uv (recommended)
uv sync --extra notebooks

# Or with pip
pip install jupyter yfinance scikit-learn
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or with JupyterLab
jupyter lab
```

Navigate to the notebooks directory and open any notebook to begin!

---

## 📖 Learning Path

### For Beginners
1. **01_basic_usage.ipynb** - Learn the basics
2. **05_commission_usage.ipynb** - Understand costs
3. **02_candle_strategies.ipynb** - Simple patterns

### For Intermediate Users
1. **03_ta_strategies.ipynb** - Technical indicators
2. **06_advanced_optimization.ipynb** - Parameter tuning
3. **02_candle_strategies.ipynb** - Pattern recognition

### For Advanced Users
1. **04_ml_strategies.ipynb** - Machine learning
2. **06_advanced_optimization.ipynb** - Advanced optimization
3. All notebooks for comprehensive understanding

---

## 💡 Tips

- **Run cells sequentially** - Notebooks build on previous cells
- **Experiment with parameters** - Change values and see what happens
- **Save your work** - Commit useful source cells; keep generated outputs stripped
- **Check data availability** - yfinance may have rate limits
- **Use different tickers** - Try your favorite stocks/ETFs

---

## 🎯 What You'll Learn

By completing all notebooks, you'll be able to:

✅ Load and prepare financial data
✅ Create custom trading strategies
✅ Backtest strategies with realistic commissions
✅ Use technical indicators effectively
✅ Apply machine learning to trading
✅ Optimize strategy parameters
✅ Compare multiple strategies
✅ Avoid common backtesting pitfalls
✅ Build production-ready trading systems

---

## 📊 Example Strategies Covered

| Strategy Type | Examples | Best For |
|--------------|----------|----------|
| **Trend Following** | Moving Average, MACD | Trending markets |
| **Mean Reversion** | Bollinger Bands, RSI | Range-bound markets |
| **Pattern Recognition** | Candlestick patterns | Short-term trading |
| **Machine Learning** | Random Forest, Gradient Boosting | Complex patterns |
| **Passive** | Buy & Hold, DCA | Long-term investing |

---

## 🔧 Troubleshooting

### Common Issues

**ModuleNotFoundError:**
```bash
# Install missing package
pip install package_name
```

**yfinance download fails:**
```python
# Try with different date range or ticker
data = yf.download("SPY", start="2020-01-01", end="2023-12-31")
```

**Jupyter not found:**
```bash
# Install Jupyter
pip install jupyter
# Or with uv
uv pip install jupyter
```

---

## 📚 Additional Resources

- **Framework Documentation**: See main README.md
- **API Reference**: See CLAUDE.md
- **GitHub Issues**: Report bugs or request features
- **Community**: Share your strategies and results

---

## 🤝 Contributing

Have a cool strategy or improvement?

1. Create a new notebook
2. Follow the existing format
3. Add clear explanations
4. Submit a pull request

---

## ⚠️ Disclaimer

**Educational purposes only!**

- Past performance ≠ future results
- Always paper trade first
- Real trading has additional costs (slippage, spreads)
- Markets can change unexpectedly
- Never risk money you can't afford to lose

---

## 🎓 Next Steps

After completing the notebooks:

1. **Build your own strategy** - Combine techniques you've learned
2. **Optimize parameters** - Find the best settings for your strategy
3. **Test on different spot instruments** - Respect the long-only, cash-funded scope
4. **Paper trade** - Test in real-time without risk
5. **Share your results** - Help others learn from your experience

Happy backtesting! 🚀📈
