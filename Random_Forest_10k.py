import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/superstore.csv', parse_dates=['Order Date'])

# Convert 'Order Date' to datetime & clean column names
df.columns = df.columns.str.strip()
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', errors='coerce')

# Set 'Order Date' as index and sort
df.set_index('Order Date', inplace=True)
df.sort_index(inplace=True)

# -----------------------------------------------------
# 1) DAILY RESAMPLING: Aggregate all orders into daily sales
# -----------------------------------------------------
sales = df[['Sales']].resample('D').sum()

# -----------------------------------------------------
# 2) CREATE LAG FEATURES
# -----------------------------------------------------
def create_lagged_features(data, lags=7):
    df_lags = data.copy()
    for i in range(1, lags + 1):
        df_lags[f'lag_{i}'] = df_lags['Sales'].shift(i)
    df_lags.dropna(inplace=True)
    return df_lags

df_lagged = create_lagged_features(sales, lags=7)

# -----------------------------------------------------
# 3) TRAIN-TEST SPLIT
# -----------------------------------------------------
train_size = int(len(df_lagged) * 0.8)
train_data = df_lagged.iloc[:train_size]
test_data  = df_lagged.iloc[train_size:]

X_train, y_train = train_data.drop(columns=['Sales']), train_data['Sales']
X_test, y_test   = test_data.drop(columns=['Sales']), test_data['Sales']

# -----------------------------------------------------
# 4) TRAIN RANDOM FOREST
# -----------------------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------
# 5) MAKE PREDICTIONS
# -----------------------------------------------------
y_pred = model.predict(X_test)

# Convert to pandas series with proper date index
y_pred_series = pd.Series(y_pred, index=y_test.index)

# -----------------------------------------------------
# 6) APPLY ROLLING AVERAGE FOR SMOOTHER LINES
# -----------------------------------------------------
rolling_window = 7  # 7-day rolling average

y_test_smooth = y_test.rolling(rolling_window).mean()
y_pred_smooth = y_pred_series.rolling(rolling_window).mean()

# -----------------------------------------------------
# 7) PLOT: SINGLE SMOOTH LINES FOR ACTUAL & PREDICTED SALES
# -----------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(y_test_smooth.index, y_test_smooth, label='Actual Daily Sales (Smoothed)',
         color='blue', linewidth=2)

plt.plot(y_pred_smooth.index, y_pred_smooth, label='Predicted Daily Sales (Smoothed)',
         color='red', linestyle='dashed', linewidth=2)

plt.title('Actual vs. Predicted Daily Sales (Smoothed)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
