import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar data sets
def load_sets():
    X_train = pd.read_csv("mlops-nvidia/data/X_train.csv", index_col=0, parse_dates=True)
    y_train = pd.read_csv("mlops-nvidia/data/y_train.csv", index_col=0, parse_dates=True).squeeze()
    X_test = pd.read_csv("mlops-nvidia/data/X_test.csv", index_col=0, parse_dates=True)
    y_test = pd.read_csv("mlops-nvidia/data/y_test.csv", index_col=0, parse_dates=True).squeeze()
    return X_train, X_test, y_train, y_test

# Entrenamiento con Random Forest
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("🌲 Random Forest entrenado correctamente")
    return model

# Evaluación
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("📊 Evaluación del modelo:")
    print(f"📈 Error absoluto medio (MAE):       {mae:.6f}")
    print(f"📉 Error cuadrático medio (MSE):     {mse:.6f}")
    print(f"🧮 Coeficiente de determinación (R²): {r2:.4f}")

# Coordinación de flujo
def main():
    X_train, X_test, y_train, y_test = load_sets()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Guardado de modelo
    os.makedirs("mlops-nvidia/models", exist_ok=True)
    joblib.dump(model, "mlops-nvidia/models/modelo_regresion.pkl")
    print("💾 Modelo guardado en models/modelo_regresion.pkl")

if __name__ == "__main__":
    main()