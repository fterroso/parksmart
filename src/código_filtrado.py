import pandas as pd

# POENMOS LOS FILTROS DEL CÓDIGO DE HACIA DONDE QUEREMOS QUE VENGAN 
CODIGO_ANTIGONES = '3001601' 
ORIGENES_VALIDOS = ['30021', '30005', '3003901'] 

archivo_entrada = 'data/20230101_Viajes_distritos.csv'  
archivo_salida = 'archivo_prueba.csv'

trozos_limpios = []

print("Iniciando la limpieza del archivo...")

# vamos leyendo los datos con pandas, usaremos chunks de 100k de datos, para evitar problemas de rendimiento

for chunk in pd.read_csv(archivo_entrada, sep='|', chunksize=100000, dtype=str):
    
    # creamos la primera mascara
    filtro_destino = chunk['destino'] == CODIGO_ANTIGONES
    
    # segunda mascara
    filtro_origen = chunk['origen'].isin(ORIGENES_VALIDOS)
    
    # nos quedamos solo con las filas que cumplan ambas mascaras
    chunk_filtrado = chunk[filtro_destino & filtro_origen]
    
    # Si después de filtrar nos ha quedado alguna fila útil, la guardamos en nuestra lista
    if not chunk_filtrado.empty:
        trozos_limpios.append(chunk_filtrado)

# exportamos el archivbo filtrado
if trozos_limpios:
    df_final_dia = pd.concat(trozos_limpios)
    
    df_final_dia.to_csv(archivo_salida, index=False) # alomejor usar otro formato como ¿.parquet?
    
    print(f"Archivo filtrado")
else:
    print("No se ha encontrado nada")