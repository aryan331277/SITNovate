import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Read and cast schema
df = pl.read_csv('/content/drive/MyDrive/retail_sales.csv')
schema = {
    "date": pl.Utf8,
    "sales": pl.Int64,
    "revenue": pl.Int64,
    "stock": pl.Int64,
    "price": pl.Float64
}
df = df.select(schema.keys()).cast(schema)

# 2. Convert date column to Date
df = df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))

# 3. Group by Date for aggregated total sales
df_grouped = df.group_by("date").agg([
    pl.sum("sales").alias("total_sales"),
])

# 4. Convert to Pandas, rename for NeuralProphet
df_pandas = df_grouped.to_pandas()
df_pandas.rename(columns={"date": "ds", "total_sales": "y"}, inplace=True)
df_pandas['ds'] = pd.to_datetime(df_pandas['ds'])

# Sort by ds for a clean line plot
df_pandas.sort_values(by='ds', inplace=True)

# 5. Train NeuralProphet
m = NeuralProphet()
df_train, df_val = m.split_df(df_pandas, freq="D", valid_p=0.2)
metrics = m.fit(df_train, freq="D", validation_df=df_val)

# 6. Generate forecast for entire historical range + 30 days future
future = m.make_future_dataframe(
    df_pandas, 
    periods=30, 
    n_historic_predictions=True  # Include all historical dates in the forecast
)
forecast = m.predict(future)

df_merged = pd.merge(
    forecast[['ds','yhat1']],
    df_pandas[['ds','y']],
    on='ds',
    how='left'
).dropna(subset=['y'])  
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

mae_all, mse_all, rmse_all, r2_all = regression_metrics(
    df_merged['y'],
    df_merged['yhat1']
)

df_train_merged = df_merged[df_merged['ds'].isin(df_train['ds'])]
mae_train, mse_train, rmse_train, r2_train = regression_metrics(
    df_train_merged['y'],
    df_train_merged['yhat1']
)

df_val_merged = df_merged[df_merged['ds'].isin(df_val['ds'])]
mae_val, mse_val, rmse_val, r2_val = regression_metrics(
    df_val_merged['y'],
    df_val_merged['yhat1']
)

print("=== Overall (All Historic) ===")
print(f"MAE:  {mae_all:.2f}")
print(f"MSE:  {mse_all:.2f}")
print(f"RMSE: {rmse_all:.2f}")
print(f"R²:   {r2_all:.4f}\n")

print("=== Training Set ===")
print(f"MAE:  {mae_train:.2f}")
print(f"MSE:  {mse_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"R²:   {r2_train:.4f}\n")

print("=== Validation Set ===")
print(f"MAE:  {mae_val:.2f}")
print(f"MSE:  {mse_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"R²:   {r2_val:.4f}")

plt.figure(figsize=(15, 7))

# Actual
plt.plot(
    df_pandas['ds'], 
    df_pandas['y'], 
    label='Actual Sales', 
    color='blue', 
    linewidth=2
)

plt.plot(
    forecast['ds'], 
    forecast['yhat1'], 
    label='Predicted Sales', 
    color='red', 
    linestyle='--', 
    linewidth=2
)

plt.title('Actual vs Predicted Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(metrics["MAE"], "-o", label="Training Loss (MAE)", linewidth=2)
ax.plot(metrics["MAE_val"], "-r", label="Validation Loss (MAE)", linewidth=2)
ax.legend(loc="center right", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=20)
ax.set_xlabel("Epoch", fontsize=28)
ax.set_ylabel("Loss (MAE)", fontsize=28)
ax.set_title("Model Loss (MAE)", fontsize=28)
plt.show()
