import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_features(filepath="mlops-nvidia/data/nvidia_features_preprocessed.csv"):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"📥 Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def split_features_and_target(df):
    X = df.drop(columns=["daily_return"])   # Features
    y = df["daily_return"].shift(-1)        # Target: retorno siguiente
    df_final = X.copy()
    df_final["target"] = y
    df_final = df_final.dropna()            # Eliminar último registro sin target
    print("✅ X e y separados. Dataset listo con shape:", df_final.shape)
    return df_final.drop(columns="target"), df_final["target"]

def normalize_features(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Guardar scaler entrenado
    os.makedirs("mlops-nvidia/models", exist_ok=True)
    joblib.dump(scaler, "mlops-nvidia/models/scaler.pkl")
    print("✅ Features normalizadas")
    print("✅ Scaler guardado en 'mlops-nvidia/models/scaler.pkl'")

    return X_scaled

def time_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"🔀 División temporal: {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test

def save_sets(X_train, X_test, y_train, y_test, path="mlops-nvidia/data/"):
    os.makedirs(path, exist_ok=True)
    X_train.to_csv(f"{path}X_train.csv")
    X_test.to_csv(f"{path}X_test.csv")
    y_train.to_csv(f"{path}y_train.csv")
    y_test.to_csv(f"{path}y_test.csv")
    print("💾 Sets de entrenamiento y prueba guardados en carpeta 'data/'")

def main():
    df = load_features()
    X, y = split_features_and_target(df)
    X_scaled = normalize_features(X)
    X_train, X_test, y_train, y_test = time_split(X_scaled, y)
    save_sets(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()