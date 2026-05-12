# ParkSmart: Análisis del uso del Aparcamiento en la UPCT en Campus la Muralla de Mar mediante Datos Abiertos

Repositorio del reto ParkSmart de la UPCT para en análisis de la movilidad de los estudiantes de la UPCT en el entorno del Campus La Muralla del Mar y un predictor de ocupación mediante el uso de los datos abiertos [Open Data Movilidad](https://www.transportes.gob.es/ministerio/proyectos-singulares/estudios-de-movilidad-con-big-data/opendata-movilidad) 

## Archivos importantes
- `src\travel_predictor_app.py`: entrena el modelo y levanta la interfaz web en HTML.
- `environment.yml`: dependencias para la creacion del entorno virtual conda.
- `calendario_murcia_cartagena_22_25.csv`: calendario académico de la UPCT para cada día entre 2022 y 2025.
- `viajes_calendario_murcia_cartagena_22_25.csv`: viajes extraidos de Open Data Movilidad.

## Uso
1. Crea un conda environment e instala dependencias:
   ```bash
   conda env create -f environment.yml
   ```
2. Lanza la app indicando tu CSV:
   ```bash
     python .\src\travel_predictor_app.py --csv .\data\viajes_calendario_murcia_cartagena_22_25.csv
   ```
3. Abre `http://127.0.0.1:5000` en el navegador.

## Formato del CSV
Debe contener, al menos, estas columnas:
- `fecha`
- `tramo_horario`
- `viajes`

## Qué hace el modelo
- Usa variables de calendario: día de la semana, mes, día del año, codificación cíclica y calendario académico
- Usa el tramo horario como categoría.
- Añade información temporal del mismo tramo: rezago 1, rezago 7 y medias móviles.
- Devuelve una estimación central y un intervalo aproximado con cuantiles.
