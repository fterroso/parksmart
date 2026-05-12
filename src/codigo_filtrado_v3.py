import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# codigos de filtro
CODIGO_ANTIGONES = '3001601' 
ORIGENES_VALIDOS = ['3003001', '3003002', '3003003', '3003004', '3003005', '3003006', '3003007', '3003008', '30027', '30005', '30035', '30036', '3003701', '3003702', '30902']

COLUMNAS_A_CONSERVAR = ['fecha', 'periodo', 'origen', 'destino', 'distancia', 'sexo', 'viajes', 'viajes_km'] 

# añadimos desde el año y mes que queremos empezar
AÑO = 2022

CARPETA_DESTINO = 'data_filtrada'

def descargar_archivo(url, nombre_destino):
    tamaño_chunk = 100 * 1024 *1024 # encontrar un equilibrio según ancho de banda de internet y ram
    print(f"Descargando: {url} ...")
    respuesta = requests.get(url, stream=True) # con stream true evitamos sobrecargar la ram
    if respuesta.status_code == 200: # comprobación de que no haya nigun error al conectarse 
        with open(nombre_destino, 'wb') as f:
            # guardamos en disco duro en bloqued de 10 MB
            for chunk in respuesta.iter_content(chunk_size=tamaño_chunk): 
                f.write(chunk)
        return True
    else:
        print(f"Error {respuesta.status_code}.")
        return False


datos_año = []

fechas_del_año = pd.date_range(start=f'{AÑO}-01-01', end = f'{AÑO}-12-31') # pandas gestiona automaticamente los bisiestos.

for fecha in fechas_del_año: # en este bucle estamos suponiendolo que hacemos paquetes de mes en mes
    # como en la web del ministerio el enlace de descarga de un dia a otro siempre tiene el formato : https://movilidad-opendata.mitma.es/estudios_basicos/por-distritos/viajes/ficheros-diarios/2022-01/20220104_Viajes_distritos.csv.gz
    
    # Nos damos cuenta que necesitamos los días con formato 01
    # Tambíen necesitaremos un parámetro que sea añomesdia, ej 20220104
    mes_str = fecha.strftime('%m')
    dia_str = fecha.strftime('%d')
    fecha_corta = fecha.strftime('%Y%m%d')
    
    url = f"https://movilidad-opendata.mitma.es/estudios_basicos/por-distritos/viajes/ficheros-diarios/{AÑO}-{mes_str}/{fecha_corta}_Viajes_distritos.csv.gz"
    archivo_temp = os.path.join(CARPETA_DESTINO, f"temp_{fecha_corta}.csv.gz") # por rendimiento del equipo, cuando acabemos de filtrar un csv, como estos suelen pesar más de 1GB, lo mejor va a ser eliminarlo y ya pasamos al siguiente, quedanos solo con los archivos reducidos
    
    if descargar_archivo(url, archivo_temp):
        print(f"Procesando {archivo_temp}...")
        trozos_limpios = []
        
        # FILTRAMOS 
        try:
            for chunk in pd.read_csv(archivo_temp, sep='|', compression='gzip', chunksize=900000, dtype=str): # nuevamente como leer 1GB al contado puede llegar a colapsar el equipo, vamos a ir de 100k filas en 100k filas
                filtro = (chunk['destino'] == CODIGO_ANTIGONES) & (chunk['origen'].isin(ORIGENES_VALIDOS))
                filtro2 = (chunk['actividad_origen'] == 'casa') & (chunk['actividad_destino'] == 'trabajo_estudio')
                filtro3 = chunk['edad'] == '0-25'
                chunk_filtrado = chunk[filtro & filtro2 & filtro3]
                
                if not chunk_filtrado.empty:
                    chunk_filtrado = chunk_filtrado[COLUMNAS_A_CONSERVAR]
                    trozos_limpios.append(chunk_filtrado)
            
            if trozos_limpios:
                datos_año.append(pd.concat(trozos_limpios))
                
        except Exception as e:
            print(f"Error leyendo el archivo {archivo_temp}: {e}")
            
        # borramos el paquete del que ya hemos filtrado la infromación que nos interesa 
        os.remove(archivo_temp)

# cuando tenemos de cada dia todas las filas que nos interesan solo queda crear un arhcivo.parquet (por rendimiento tal vez un csv puede funcionar bien)
if datos_año:
    df_año_completo = pd.concat(datos_año)
    nombre_parquet = os.path.join(CARPETA_DESTINO, f"viajes_murcia_cartagena_{AÑO}.parquet")
    
    df_año_completo.to_parquet(nombre_parquet, engine='pyarrow')
    print(f"Año completado Guardado en: {nombre_parquet}")
else:
    print("No hay datos útiles en todo el mes.")
    
    
    
    
    
    




