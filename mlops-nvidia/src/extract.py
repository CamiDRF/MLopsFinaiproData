# Extraccion de datos 
import yfinance as yf
import pandas as pd

def fetch_nvidia_data():
    nvda = yf.Ticker("NVDA")
    df = nvda.history(period="10y")
    df.to_csv("mlops-nvidia/data/nvidia_data.csv")
    print("Extracción Exitosa de Datos de NVIDIA y guardados como CSV.")
    
if __name__ == "__main__":
    fetch_nvidia_data()
