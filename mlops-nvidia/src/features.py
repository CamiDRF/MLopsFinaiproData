import pandas as pd
import matplotlib.pyplot as plt
from skimpy import skim
from sklearn.preprocessing import StandardScaler

# 📥 Cargar datos originales
df = pd.read_csv("mlops-nvidia/data/nvidia_data.csv", index_col=0, parse_dates=True)

# 🧪 Análisis exploratorio
print("\n--- Info ---")
print(df.info())
print("\n--- Head ---")
print(df.head())
print("\n--- Estadísticas ---")
print(skim(df))
print("\n--- Datos nulos ---")
print(df.isnull().sum())

# 🧹 Limpieza de columnas irrelevantes
df_clean = df.copy()
df_clean = df_clean.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
print("Columnas después de limpieza:", df_clean.columns)

# 🎯 Ingeniería de características
df_features = df_clean.copy()

# Retorno diario
df_features["daily_return"] = df_features["Close"].pct_change()

# Media móvil de 20 días
df_features["ma_20"] = df_features["Close"].rolling(window=20).mean()

# Volatilidad (desviación estándar de 20 días)
df_features["volatility"] = df_features["Close"].rolling(window=20).std()

# Relación High / Low
df_features["hl_ratio"] = df_features["High"] / df_features["Low"]

# Z-score del volumen
df_features["volume_zscore"] = (
    df_features["Volume"] - df_features["Volume"].mean()
) / df_features["Volume"].std()

# 🔁 Lags del retorno
df_features["return_lag_1"] = df_features["daily_return"].shift(1)
df_features["return_lag_2"] = df_features["daily_return"].shift(2)

# 📈 Relación precio vs media móvil
df_features["price_vs_ma20"] = df_features["Close"] / df_features["ma_20"] - 1

# 📅 Codificación del día de la semana
df_features.index = pd.to_datetime(df_features.index, utc=True).tz_convert(None)
df_features["weekday"] = df_features.index.dayofweek
df_features = pd.get_dummies(df_features, columns=["weekday"], prefix="wd", drop_first=True)

# 🧼 Limpiar valores nulos
df_features = df_features.dropna()

# 🔍 Revisión final
print(df_features.head())
print("Variables generadas:", df_features.columns.tolist())

# 💾 Guardar dataset enriquecido
df_features.to_csv("mlops-nvidia/data/nvidia_features_preprocessed.csv")
print("✅ Dataset enriquecido guardado como 'nvidia_features_preprocessed.csv'")