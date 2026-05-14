import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set visual style
sns.set_theme(style="whitegrid")

# =================================================================
# 1. DATA LOADING & PREPROCESSING
# =================================================================
df = pd.read_csv('GlobalWeatherRepository.csv')
df['last_updated'] = pd.to_datetime(df['last_updated'])
df['date'] = df['last_updated'].dt.date

# Handle Air Quality placeholders (-9999) and extreme sensor errors
aq_cols = [col for col in df.columns if 'air_quality' in col]
for col in aq_cols:
    df.loc[df[col] < -500, col] = np.nan

# Handle extreme wind outliers (likely errors)
df.loc[df['wind_mph'] > 250, 'wind_mph'] = np.nan
df.loc[df['wind_kph'] > 400, 'wind_kph'] = np.nan

# Simple imputation using median for numeric columns
df = df.fillna(df.median(numeric_only=True))

# =================================================================
# 2. ADVANCED EDA: ANOMALY DETECTION
# =================================================================
# Identifying unusual weather patterns using Isolation Forest
features_for_anomaly = df[['temperature_celsius', 'wind_mph', 'humidity', 'pressure_mb']]
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = iso_forest.fit_predict(features_for_anomaly)

# Export detected anomalies
anomalies_df = df[df['anomaly'] == -1]
anomalies_df.to_csv('anomalies_detected.csv', index=False)
print(f"Anomalies identified and saved: {len(anomalies_df)} records.")

# =================================================================
# 3. FEATURE ENGINEERING & TIME-SERIES PREPARATION
# =================================================================
# Aggregate to global daily averages
daily_data = df.groupby('date')['temperature_celsius'].mean().reset_index()
daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data = daily_data.sort_values('date')

# Create lagging and rolling features
daily_data['lag_1'] = daily_data['temperature_celsius'].shift(1)
daily_data['lag_7'] = daily_data['temperature_celsius'].shift(7)
daily_data['rolling_mean_7'] = daily_data['temperature_celsius'].rolling(window=7).mean()
daily_data['month'] = daily_data['date'].dt.month
daily_data['day_of_year'] = daily_data['date'].dt.dayofyear

daily_data = daily_data.dropna()

# =================================================================
# 4. MULTI-MODEL FORECASTING & EVALUATION
# =================================================================
X = daily_data[['lag_1', 'lag_7', 'rolling_mean_7', 'month', 'day_of_year']]
y = daily_data['temperature_celsius']

# Sequential Split (80% Train, 20% Test)
split_idx = int(len(daily_data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Model 1: Linear Regression
lr_model = LinearRegression().fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Model 3: Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

# Model 4: Ensemble (Averaging predictions)
ensemble_preds = (rf_preds + gb_preds) / 2

# Calculate Metrics
model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Ensemble']
preds_list = [lr_preds, rf_preds, gb_preds, ensemble_preds]

perf_data = []
for name, p in zip(model_names, preds_list):
    rmse = np.sqrt(mean_squared_error(y_test, p))
    mae = mean_absolute_error(y_test, p)
    perf_data.append({'Model': name, 'RMSE': rmse, 'MAE': mae})

performance_df = pd.DataFrame(perf_data)
performance_df.to_csv('model_performance.csv', index=False)
print("Model performance evaluation completed and saved.")

# =================================================================
# 5. VISUALIZATION GENERATION
# =================================================================

# Plot 1: Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# Plot 2: Distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['temperature_celsius'], bins=30, kde=True, color='orange', ax=axes[0])
axes[0].set_title('Global Temperature Distribution')
sns.histplot(df['precip_mm'], bins=30, kde=True, color='blue', ax=axes[1])
axes[1].set_title('Global Precipitation Distribution (Log Scale)')
axes[1].set_yscale('log')
plt.savefig('temp_precip_dist.png')

# Plot 3: Daily Global Trend
plt.figure(figsize=(12, 6))
plt.plot(daily_data['date'], daily_data['temperature_celsius'], color='green')
plt.title('Global Average Daily Temperature (2024-2026)')
plt.xticks(rotation=45)
plt.savefig('global_temp_trend.png')

# Plot 4: Forecast Validation
plt.figure(figsize=(14, 7))
test_dates = daily_data['date'].iloc[split_idx:]
plt.plot(test_dates, y_test, label='Actual', color='black', lw=2)
plt.plot(test_dates, ensemble_preds, label='Ensemble Forecast', color='red', ls='--')
plt.title('Weather Trend Forecast: Actual vs Predicted')
plt.legend()
plt.savefig('forecast_results.png')

plt.show()