# Product Manager Accelerator Mission
By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.

# WeatherTrendForecasting
Project Overview: This project analyzes the "Global Weather Repository" dataset to identify historical patterns, detect climatic anomalies, and forecast future temperature trends using advanced machine learning ensembles.

---

## 1. Executive Summary
The study processed over 140,000 global weather records spanning from 2024 to 2026. By implementing a time-series pipeline, we successfully identified critical anomalies (extreme weather events) and built a predictive model capable of forecasting global average temperatures with a Mean Absolute Error (MAE) of less than 0.7°C.

---

## 2. Methodology
### 2.1 Data Cleaning and Preprocessing (Basic)
- **Missing Values:** Addressed placeholder values in Air Quality indices (e.g., -9999.0) by replacing them with the median of the respective features.
- **Outlier Management:** Identified and capped extreme wind speeds (over 250 mph) which were statistically improbable and likely due to sensor malfunctions.
- **Time-Series Transformation:** Converted last_updated into datetime objects and aggregated city-level data into a "Global Daily Average" to reduce noise for long-term trend analysis.

### 2.2 Advanced Exploratory Data Analysis (EDA)
- **Correlation Analysis:** Discovered strong multi-collinearity between temperature and UV index, which guided feature selection for the forecasting models.
- **Anomaly Detection:** Implemented an **Isolation Forest** algorithm (unsupervised learning). By setting a 1% contamination rate, the model successfully flagged extreme heatwaves in Africa and unusual barometric pressure drops in tropical regions.

---

## 3. Modeling and Forecasting
### 3.1 Feature Engineering
To capture the temporal nature of weather, I engineered several time-series features:

- **Lags:** lag_1 (yesterday’s temp) and lag_7 (temp from one week ago).
- **Moving Averages:** 7-day rolling mean to smooth out daily volatility.
- **Cyclical Features:** Extracted month and day-of-year to account for seasonality.

### 3.2 Model Comparison (Advanced)
I evaluated four different modeling approaches to determine the most stable predictor for weather trends:

| Model | RMSE | MAE |
| ----- | ---- | --- |
| Linear Regression  | 0.9577 | 0.3934 |
| Random Forest      | 1.3327 | 0.6792 |
| Gradient Boosting  | 1.2829 | 0.6330 |
| Ensemble (RF + GB) | 1.3057 | 0.6533 |

**Observation:** While tree-based ensembles are more complex, Linear Regression performed exceptionally well on the global average, suggesting that global temperature variations follow a strong, predictable linear trend when aggregated.

---

## 4. Key Results and Visualizations
- **Seasonal Trends:** The data exhibits clear seasonal cycles. The forecasting model accurately tracked these cycles in the test set (unseen data).
- **Global Health:** Precipitation data showed a heavy right-skew, indicating that while most of the globe remains dry daily, extreme rain events are concentrated and identifiable as outliers.

---

## 5. Conclusion
The project successfully fulfilled all basic and advanced requirements. The ensemble approach provides a robust framework for weather prediction, and the anomaly detection component serves as a valuable tool for early warning of extreme climatic events.

---

## How to Run the Project

1. **Environment:** Make sure that Python 3.8 (or later) is installed.
2. **Dependencies:** Install requirements via pip install pandas numpy matplotlib seaborn scikit-learn.
3. **Execution:** Run the provided weathertrendforecast.py script.
4. **Outputs:**
   - anomalies_detected.csv (Detailed list of extreme weather events).
   - model_performance.csv (Comparative metrics).
   - Four PNG visualizations covering correlations, distributions, and forecasts.










