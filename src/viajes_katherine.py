from pathlib import Path
import re
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
# =========================
# Configuración
# =========================
PATRON_FICHEROS = "data/viajes_murcia_cartagena_*.parquet"
SOLO_LABORABLES = True
SOLO_TEMPORADA_ESCOLAR = True  # Sept-Jun (aprox., sin festivos específicos)
def extraer_ano_desde_nombre(path: Path) -> int:
    m = re.search(r"(\d{4})", path.stem)
    if not m:
        raise ValueError(f"No se pudo extraer año de {path.name}")
    return int(m.group(1))
def parsear_viajes(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )
def cargar_y_preparar(path: Path) -> pd.DataFrame:
    ano = extraer_ano_desde_nombre(path)
    df = pd.read_parquet(path).copy()
    df["ano"] = ano
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y%m%d", errors="coerce")
    df["periodo"] = pd.to_numeric(df["periodo"], errors="coerce")
    df["viajes"] = parsear_viajes(df["viajes"])
    # Limpieza
    df = df.dropna(subset=["fecha", "periodo", "viajes"])
    df["periodo"] = df["periodo"].astype(int)
    # Día semana
    mapa_dias = {
        0: "Lunes", 1: "Martes", 2: "Miércoles",
        3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"
    }
    df["dia_num"] = df["fecha"].dt.weekday
    df["dia_semana"] = df["dia_num"].map(mapa_dias)
    # Filtros
    if SOLO_LABORABLES:
        df = df[df["dia_num"] < 5]
    if SOLO_TEMPORADA_ESCOLAR:
        # Aproximación lectivo: Sept-Jun (excluye Jul/Ago)
        df = df[df["fecha"].dt.month.isin([9, 10, 11, 12, 1, 2, 3, 4, 5, 6])]
    return df
def main():
    rutas = sorted(Path(".").glob(PATRON_FICHEROS))
    if not rutas:
        raise FileNotFoundError(f"No se encontraron ficheros con patrón: {PATRON_FICHEROS}")
    df = pd.concat([cargar_y_preparar(r) for r in rutas], ignore_index=True)
    # =========================
    # 1) Perfil horario por año (línea)
    # =========================
    perfil = (
        df.groupby(["ano", "periodo"], as_index=False)["viajes"]
        .sum()
        .sort_values(["ano", "periodo"])
    )
    fig1 = px.line(
        perfil,
        x="periodo",
        y="viajes",
        color="ano",
        markers=True,
        title="Perfil horario comparado por año (laborables, temporada escolar)",
        labels={"periodo": "Franja horaria (periodo)", "viajes": "Viajes", "ano": "Año"},
    )
    fig1.update_layout(hovermode="x unified")
    fig1.show()
    # =========================
    # 2) Heatmap día laboral x franja (animation por año)
    # =========================
    heat = (
        df.groupby(["ano", "dia_semana", "periodo"], as_index=False)["viajes"]
        .sum()
    )
    orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
    heat["dia_semana"] = pd.Categorical(heat["dia_semana"], categories=orden_dias, ordered=True)
    heat = heat.sort_values(["ano", "dia_semana", "periodo"])
    fig2 = px.density_heatmap(
        heat,
        x="periodo",
        y="dia_semana",
        z="viajes",
        animation_frame="ano",
        color_continuous_scale="YlOrRd",
        title="Heatmap día laboral × franja (por año)",
        labels={"periodo": "Franja", "dia_semana": "Día", "viajes": "Viajes"},
    )
    fig2.show()
    # =========================
    # 3) Top 15 rutas O-D (barras)
    # =========================
    mapa_zonas = {}
    with open("data/IDS_distritos.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                mapa_zonas[k.strip()] = v.strip()

    mapa_zonas.setdefault("3001601", "Cartagena distrito 01")
    # 2) Enriquecer columnas
    df["origen_cod"] = df["origen"].astype(str)
    df["destino_cod"] = df["destino"].astype(str)
    df["origen_nombre"] = df["origen_cod"].map(mapa_zonas).fillna("Código " + df["origen_cod"])
    df["destino_nombre"] = df["destino_cod"].map(mapa_zonas).fillna("Código " + df["destino_cod"])
    df["origen_lbl"] = df["origen_nombre"] + " (" + df["origen_cod"] + ")"
    df["destino_lbl"] = df["destino_nombre"] + " (" + df["destino_cod"] + ")"
    df["ruta"] = df["origen_lbl"] + " → " + df["destino_lbl"]
    # 3) Agregación por año y ruta
    base = (
        df.groupby(["ano", "ruta", "origen_nombre", "origen_cod", "destino_nombre", "destino_cod"], as_index=False)["viajes"]
        .sum()
    )
    # 4) Tomar Top15 dentro de cada año
    top15_por_ano = (
        base.sort_values(["ano", "viajes"], ascending=[True, False])
            .groupby("ano", as_index=False)
            .head(15)
            .copy()
    )
    # ranking + porcentajes dentro de cada año
    top15_por_ano["ranking"] = top15_por_ano.groupby("ano")["viajes"].rank(method="first", ascending=False).astype(int)
    totales_ano = top15_por_ano.groupby("ano")["viajes"].transform("sum")
    top15_por_ano["pct_top15_ano"] = (top15_por_ano["viajes"] / totales_ano) * 100
    totales_dataset_ano = base.groupby("ano")["viajes"].sum().rename("total_ano").reset_index()
    top15_por_ano = top15_por_ano.merge(totales_dataset_ano, on="ano", how="left")
    top15_por_ano["pct_total_ano"] = (top15_por_ano["viajes"] / top15_por_ano["total_ano"]) * 100
    # 5) Gráfico animado
    fig = px.bar(
        top15_por_ano.sort_values(["ano", "viajes"]),
        x="viajes",
        y="ruta",
        orientation="h",
        animation_frame="ano",
        title="Top 15 rutas Origen → Destino por año",
        labels={"viajes": "Viajes", "ruta": "Ruta", "ano": "Año"},
        custom_data=[
            "ranking", "origen_nombre", "origen_cod",
            "destino_nombre", "destino_cod",
            "pct_top15_ano", "pct_total_ano"
        ]
    )
    fig.update_traces(
        hovertemplate=(
            "<b>Ranking:</b> Top %{customdata[0]}<br>"
            "<b>Ruta:</b> %{y}<br>"
            "<b>Origen:</b> %{customdata[1]} (%{customdata[2]})<br>"
            "<b>Destino:</b> %{customdata[3]} (%{customdata[4]})<br>"
            "<b>Viajes:</b> %{x:,.2f}<br>"
            "<b>% dentro Top15 (año):</b> %{customdata[5]:.2f}%<br>"
            "<b>% sobre total del año:</b> %{customdata[6]:.2f}%<extra></extra>"
        )
    )   
    fig.show()
if __name__ == "__main__":
    main()