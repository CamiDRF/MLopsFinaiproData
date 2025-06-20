import pandas as pd
import joblib

# 📥 Cargar modelo y scaler
model = joblib.load("mlops-nvidia/models/modelo_regresion.pkl")
scaler = joblib.load("mlops-nvidia/models/scaler.pkl")

# 📊 Cargar datos de features preprocesadas
df = pd.read_csv("mlops-nvidia/data/nvidia_features_preprocessed.csv", index_col=0, parse_dates=True)

# 🔍 Tomar el último día disponible (sin el target)
X_new = df.tail(1).drop(columns=["daily_return"])

# 🧪 Aplicar normalización
X_new_scaled = scaler.transform(X_new)

# 🔮 Realizar predicción
prediccion = model.predict(X_new_scaled)[0]

# 🖨️ Mostrar resultados
print("\n🔍 Últimas features utilizadas para inferencia:")
print(X_new.T)

print(f"\n🔮 Retorno estimado para el próximo día: {prediccion:.4%}")