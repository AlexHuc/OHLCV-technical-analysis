# OHLCV Stock Analysis - Technical Chart Labeling Dataset

Synthetically generated candlestick charts with human-annotated technical analysis labels for machine learning training.

---

**Data Source**: Synthetically Generated with Human Technical Analysis Annotations

## About this Dataset

This dataset provides **1,219 JSON files** containing synthetically generated OHLCV (Open-High-Low-Close-Volume) candlestick data with meticulous human-provided technical analysis labels. Each file represents a complete 300-candle chart window, making it ideal for training machine learning models on automated technical analysis, pattern recognition, and chart-based trading signal generation.

The dataset bridges the gap between synthetic data availability and real-world trading analysis needs by combining structurally plausible candlestick patterns with expert human annotations identifying key technical levels and formations.

## How to Use the Dataset

### 1. Understand Your Research Question
- Do you want to predict **price direction** (up/down movement)?
- Are you interested in **price target forecasting** (regression)?
- Do you want to **identify support/resistance levels** and trendlines?
- Are you building a **chart pattern recognition model** (CNN/Vision Transformer)?

### 2. Data Exploration & Feature Engineering
- Load JSON files and extract OHLCV sequences
- Engineer technical indicators: volatility, momentum, trend slopes, moving averages
- Analyze human-provided labels (horizontal_lines, ray_lines) to understand labeling patterns
- Identify and handle edge cases or outliers in synthetic data generation

### 3. Account for Data Characteristics
- **Synthetic Nature**: Price and volume values do not represent real financial instruments
- **Realistic Structure**: Candlestick patterns maintain realistic intra-day time progression
- **Temporal Metadata**: Timestamps follow ISO 8601 format but have synthetic dates
- **Human Bias**: Labels reflect individual analyst interpretations of technical analysis
- **Consistency**: Focus on relative price structure and proportions rather than absolute values

### 4. Model Development Approaches
- **Time Series Classification**: Predict next candle direction using OHLCV history
- **Price Regression**: Forecast exact closing prices
- **Pattern Recognition**: Identify chart formations using CNNs on candlestick visualizations
- **Level Detection**: Predict support/resistance zones from ray_lines labels
- **Ensemble Methods**: Combine classification + regression for trading signals

## Important Considerations

⚠️ **Data Limitations**:
- Purely **synthetic data** - not suitable for live trading without additional validation
- Timestamps are **artificial** - do not reflect real market dates
- Prices/volumes are **not tied to real financial instruments**
- Use for **research, backtesting simulation, and model development only**

✅ **Best Practices**:
- Treat as a **proof-of-concept dataset** for algorithm development
- Validate findings on real market data before deployment
- Focus on **pattern structure** rather than absolute numerical values
- Combine with real data for production trading systems

## Research Ideas

1. **Technical Pattern Recognition**: Train CNNs to automatically identify support/resistance levels and trendlines from chart images, comparing against human labels
2. **Price Direction Prediction**: Build XGBoost classification models using OHLCV-derived features to predict next candle direction with confidence scores
3. **Support/Resistance Forecasting**: Use regression models to predict future price levels where the market might reverse
4. **Label Analysis**: Study human analyst labeling patterns and consistency - which levels are most commonly identified?
5. **Data Augmentation**: Generate synthetic variations of labeled charts to expand training dataset
6. **Multi-timeframe Analysis**: Correlate patterns across different chart windows to identify hierarchical market structures
7. **Transformer Models**: Implement attention-based models to capture temporal dependencies in candlestick sequences

### JSON File Schema

Each JSON file contains:

```json
{
  "metadata": {
    "generation_timestamp": "2025-01-05T14:32:18Z",
    "version": "1.0",
    "chart_id": "candlestick_00001"
  },
  "ohlcv_data": [
    {
      "time": "2025-01-05T09:00:00Z",
      "open": 100.50,
      "high": 102.75,
      "low": 100.25,
      "close": 102.00,
      "volume": 1500.0
    },
    ...  // 300 candles total
  ],
  "labels": {
    "horizontal_lines": [
      { "price": 102.50, "label": "resistance" },
      { "price": 99.75, "label": "support" }
    ],
    "ray_lines": [
      {
        "start_date": "2025-01-01T09:00:00Z",
        "start_price": 100.00,
        "end_date": "2025-01-05T16:00:00Z",
        "end_price": 102.50,
        "label": "trendline_up"
      }
    ]
  }
}
```

## Key Columns

| Element | Type | Description |
|---------|------|-------------|
| `time` | STRING | ISO 8601 timestamp with realistic intra-day progression |
| `open` | FLOAT64 | Synthetic opening price for the candle |
| `high` | FLOAT64 | Synthetic highest price during the period |
| `low` | FLOAT64 | Synthetic lowest price during the period |
| `close` | FLOAT64 | Synthetic closing price for the candle |
| `volume` | FLOAT64 | Synthetic trading volume during the period |
| `horizontal_lines.price` | FLOAT64 | Support/resistance level identified by analyst |
| `ray_lines.start_price` | FLOAT64 | Starting price of identified trendline |
| `ray_lines.end_price` | FLOAT64 | Ending price of identified trendline |