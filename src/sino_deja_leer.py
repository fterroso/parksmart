
import pandas as pd
# Intenta leerlo así
try:
    df = pd.read_parquet('../data_filtrada/viajes_upct_2025.parquet', engine='fastparquet')
    print("¡Logré leerlo con fastparquet!")
except Exception as e:
    print(f"Ni con esas... Error: {e}")