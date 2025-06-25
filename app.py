# app.py (Versión Final Optimizada)

import streamlit as st
import pandas as pd
import numpy as np
# Ya no necesitamos joblib, scikit-learn ni sentence-transformers aquí,
# porque todo el trabajo pesado ya está hecho.

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Ikigai-ML: Tu Orientador Profesional",
    page_icon="🤖",
    layout="centered"
)

# --- Carga de Datos (Archivos ya procesados y ligeros) ---
@st.cache_data
def cargar_datos_finales():
    data_path = Path("./data")
    try:
        # Cargamos el PUENTE INTELIGENTE que tiene todas las coincidencias
        df_puente = pd.read_parquet(data_path / "puente_onet_dane_ia.parquet")
        
        # Cargamos los datos del DANE para el conteo de frecuencia
        df_dane = pd.read_parquet(data_path / "dane_enriquecido_final_2024.parquet")
        
        # Cargamos los títulos de O*NET para tener la lista de opciones
        df_onet_titulos = pd.read_parquet(data_path / "onet_titles.parquet")
        
    except FileNotFoundError as e:
        st.error(f"Error al cargar un archivo necesario: {e}. Asegúrate de que todos los archivos .parquet estén en la carpeta 'data'.")
        return None, None, None

    return df_puente, df_dane, df_onet_titulos

# Cargamos los activos
df_puente, df_dane, df_onet_titulos = cargar_datos_finales()

# --- Interfaz de Usuario ---
st.title("🤖 Proyecto Ikigai-ML")
st.header("Tu Orientador Profesional con IA")
st.write("Selecciona una profesión de la lista para ver su perfil y su contexto en el mercado laboral colombiano.")

if df_puente is not None:
    # Creamos la lista de profesiones de O*NET para el desplegable
    lista_profesiones_onet = sorted(df_puente['Onet_Title'].unique())

    # Widget de selección para que el usuario elija una profesión
    profesion_seleccionada = st.selectbox(
        "Elige una profesión para analizar:",
        options=lista_profesiones_onet,
        index=None, # Para que no haya nada seleccionado al principio
        placeholder="Busca y selecciona una profesión..."
    )

    # Si el usuario selecciona una profesión, mostramos la información
    if profesion_seleccionada:
        
        # Buscamos la información en nuestro puente (búsqueda súper rápida)
        info_completa = df_puente[df_puente['Onet_Title'] == profesion_seleccionada]
        
        if not info_completa.empty:
            # Extraemos la información pre-calculada
            nombre_dane = info_completa['Dane_Name'].iloc[0]
            descripcion_dane = info_completa['Dane_Description'].iloc[0]
            similitud = info_completa['Similarity_Score'].iloc[0]
            
            # Buscamos el conteo en los datos del DANE
            conteo = len(df_dane[df_dane['Nombre Ocupación'] == nombre_dane])

            st.markdown(f"### {profesion_seleccionada}")
            st.info(f"**Ocupación Semánticamente Similar en Colombia:** {nombre_dane}")
            st.progress(int(similitud * 100), text=f"Puntuación de Similitud: {similitud:.2f}")
            
            st.metric(label="Presencia en la Encuesta Nacional DANE 2024", value=f"{conteo:,}".replace(',', '.'))
            
            with st.expander("Ver Descripción Completa del Perfil en Colombia (DANE)"):
                st.write(descripcion_dane)
else:
    st.error("La aplicación no pudo cargar los datos necesarios para funcionar.")