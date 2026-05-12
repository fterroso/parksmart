import pandas as pd
import requests
import os
import tarfile
import shutil
import time 

# codigos de filtro
CODIGO_ANTIGONES = '3001601' 

DESTINOS_VALIDOS = ['3003001', '3003002', '3003003', '3003004', '3003005', '3003006', '3003007', '3003008', '30027', '30005', '30035', '30036', '3003701', '3003702', '30902']

COLUMNAS_A_CONSERVAR = ['fecha', 'periodo', 'origen', 'destino', 'distancia', 'sexo', 'viajes', 'viajes_km'] 

# añadimos desde el año y mes que queremos empezar
AÑO = 2025

CARPETA_DESTINO = 'data_filtrada'



def descargar_archivo(url, nombre_destino):
    # Encontrar un equilibrio según ancho de banda de internet y ram
    tamaño_chunk = 100 * 1024 * 1024 
    print(f"\nDescargando: {url} ...")
    respuesta = requests.get(url, stream=True) 
    
    if respuesta.status_code == 200: 
        with open(nombre_destino, 'wb') as f:
            for chunk in respuesta.iter_content(chunk_size=tamaño_chunk): 
                f.write(chunk)
        return True
    else:
        print(f"Error {respuesta.status_code}.")
        return False

datos_año = []

# Iteramos del mes 1 al 12
for mes in range(1, 13): 
    # Formateamos el mes a dos dígitos (ej. "01" en vez de "1")
    mes_str = f"{mes:02d}"
    añomes = f"{AÑO}{mes_str}"
    
    # URL y rutas temporales adaptadas al formato .tar
    url = f"https://movilidad-opendata.mitma.es/estudios_basicos/por-distritos/viajes/meses-completos/{añomes}_Viajes_distritos.tar"
    archivo_tar = os.path.join(CARPETA_DESTINO, f"temp_mes_{añomes}.tar")
    carpeta_ext = os.path.join(CARPETA_DESTINO, f"temp_extraccion_{añomes}")
    
    if descargar_archivo(url, archivo_tar):
        print(f"Extrayendo archivos de {archivo_tar}...")
        
        # Creamos la carpeta de extracción si no existe
        os.makedirs(carpeta_ext, exist_ok=True)
        
        try:
            # 1. ABRIMOS LA CAJA .tar Y EXTRAEMOS TODO EN LA CARPETA
            with tarfile.open(archivo_tar, "r:*") as tar:
                tar.extractall(path=carpeta_ext)
                
            # 2. BUSCAMOS Y FILTRAMOS CADA ARCHIVO EXTRAÍDO
            # Usamos os.walk por si el TAR trae los archivos metidos en subcarpetas
            for root, dirs, files in os.walk(carpeta_ext):
                for file in files:
                    if file.endswith('.csv') or file.endswith('.csv.gz'):
                        archivo_diario = os.path.join(root, file)
                        print(f"  Procesando día: {file}...")
                        
                        trozos_limpios = []
                        
                        # FILTRAMOS con chunksize
                        # AHORA:
                        with pd.read_csv(archivo_diario, sep='|', compression='infer', chunksize=900000, dtype=str) as reader:
                            for chunk in reader:
                                filtro = (chunk['destino'].isin(DESTINOS_VALIDOS)) & (chunk['origen'] == CODIGO_ANTIGONES)
                                filtro2 = (chunk['actividad_origen'] == 'trabajo_estudio') & (chunk['actividad_destino'] == 'casa')
                                filtro3 = chunk['edad'] == '0-25'
                                
                                chunk_filtrado = chunk[filtro & filtro2 & filtro3]
                                
                                if not chunk_filtrado.empty:
                                    chunk_filtrado = chunk_filtrado[COLUMNAS_A_CONSERVAR]
                                    trozos_limpios.append(chunk_filtrado)
                        
                        if trozos_limpios:
                            datos_año.append(pd.concat(trozos_limpios))
                            
        except Exception as e:
            print(f"Error procesando el mes {añomes}: {e}")
            
        # ==========================================
        # NUEVO: LIMPIEZA EXTREMA ANTI-ERRORES WINDOWS
        # ==========================================
        if os.path.exists(carpeta_ext):
            time.sleep(1.5) # Respiro para el antivirus de Windows
            shutil.rmtree(carpeta_ext, ignore_errors=True) # ignore_errors=True es mano de santo
            
        if os.path.exists(archivo_tar):
            try:
                os.remove(archivo_tar)
            except Exception as e:
                print(f"Aviso: El .tar se quedó bloqueado y no se borró ({e})")
                
        print(f"Basura temporal del mes {añomes} eliminada correctamente.")

# ==========================================
# EXPORTACIÓN ANUAL
# ==========================================
if datos_año:
    df_año_completo = pd.concat(datos_año)
    nombre_parquet = os.path.join(CARPETA_DESTINO, f"viajes_cartagena_murcia_{AÑO}.parquet")
    
    df_año_completo.to_parquet(nombre_parquet, engine='pyarrow')
    print(f"\n¡AÑO COMPLETADO! Guardado en: {nombre_parquet}")
else:
    print("\nNo hay datos útiles en todo el año.")
    
    
