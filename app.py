# app.py (Versión Final Profesional y Multilingüe)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Ikigai-ML: Tu Orientador Profesional",
    page_icon="🤖",
    layout="wide"
)

# --- Carga de Datos y Modelos (con caché para alta velocidad) ---
@st.cache_data
def cargar_activos():
    """Carga todos los archivos necesarios para la aplicación una sola vez."""
    data_path = Path("./data")
    try:
        # Cargar el recomendador O*NET
        knn_model = joblib.load(data_path / "knn_model.pkl")
        pivot_onet = pd.read_parquet(data_path / "mat_full.parquet")
        df_onet_titulos = pd.read_parquet(data_path / "onet_titles.parquet")
        
        # Cargar el puente inteligente que conecta O*NET y DANE
        df_puente = pd.read_parquet(data_path / "puente_onet_dane_ia.parquet")
        
        # Cargar los datos del DANE para el conteo de frecuencia
        df_dane = pd.read_parquet(data_path / "dane_enriquecido_final_2024.parquet")
        
        # Cargar el archivo de traducción de habilidades
        df_traducciones = pd.read_csv(data_path / "habilidades_traduccion.csv")

    except FileNotFoundError as e:
        st.error(f"Error al cargar un archivo necesario: {e}. Asegúrate de que todos los archivos estén en la carpeta 'data'.")
        return None, None, None, None, None, None

    return knn_model, pivot_onet, df_onet_titulos, df_puente, df_dane, df_traducciones

# Cargamos los activos al iniciar la app
knn_model, pivot_onet, df_onet_titulos, df_puente, df_dane, df_traducciones = cargar_activos()


# --- Función de Vectorización (del notebook original) ---
def vector_usuario(user_skills, reference_matrix):
    """Convierte una lista de habilidades en un vector numérico para el modelo."""
    vector = np.zeros((1, len(reference_matrix.columns)))
    skills_encontradas = [skill for skill in user_skills if skill in reference_matrix.columns]
    
    if skills_encontradas:
        for skill in skills_encontradas:
            vector[0, reference_matrix.columns.get_loc(skill)] = 1
        return vector / vector.sum()
    return vector


# --- Interfaz de Usuario de la Aplicación ---
st.title("🤖 Proyecto Ikigai-ML")
st.header("Tu Orientador Profesional con Inteligencia Artificial")
st.write(
    "Descubre qué profesiones se ajustan a tus habilidades y conoce su contexto en el mercado laboral colombiano. "
    "Este proyecto combina datos de O*NET (EE.UU.) con la Gran Encuesta Integrada de Hogares del DANE (Colombia 2024)."
)
st.markdown("---")


# Comprobamos que los datos se cargaron antes de mostrar la UI
if pivot_onet is not None and df_traducciones is not None:
    st.subheader("Paso 1: Selecciona tus Habilidades")

    # Creamos un diccionario para el widget de selección: {'Español': 'English'}
    mapa_es_a_en = pd.Series(df_traducciones.skill_en.values, index=df_traducciones.skill_es).to_dict()
    opciones_habilidades_es = sorted(mapa_es_a_en.keys())

    # --- WIDGET DE SELECCIÓN MÚLTIPLE EN ESPAÑOL ---
    habilidades_seleccionadas_es = st.multiselect(
        "Selecciona tus habilidades de la lista (puedes escribir para buscar):",
        options=opciones_habilidades_es,
        placeholder="Elige una o varias habilidades"
    )

    st.markdown("---")

    # Botón para ejecutar el análisis
    if st.button("Encontrar mi Ikigai ✨"):
        if habilidades_seleccionadas_es:
            # Traducimos las habilidades seleccionadas al inglés para el modelo
            habilidades_en_ingles = [mapa_es_a_en[skill_es] for skill_es in habilidades_seleccionadas_es]
            st.info(f"Habilidades seleccionadas (en inglés para el modelo): {', '.join(habilidades_en_ingles)}")
            
            # --- 1. Generar Recomendaciones de O*NET ---
            st.subheader("Paso 2: Tus Profesiones Recomendadas")
            
            u_vec = vector_usuario(habilidades_en_ingles, pivot_onet)
            
            # Usamos el modelo k-NN para encontrar las 5 profesiones más cercanas
            distances, indices = knn_model.kneighbors(u_vec, n_neighbors=5)
            
            onet_results_titulos = df_onet_titulos.iloc[indices[0]]

            # --- 2. Enriquecer y Mostrar los Resultados ---
            for index, row in onet_results_titulos.iterrows():
                titulo_onet = row['Title']
                
                st.markdown(f"#### {titulo_onet}")
                
                info_local = df_puente[df_puente['Onet_Title'] == titulo_onet]
                
                if not info_local.empty:
                    nombre_dane = info_local['Dane_Name'].iloc[0]
                    descripcion_dane = info_local['Dane_Description'].iloc[0]
                    similitud = info_local['Similarity_Score'].iloc[0]
                    
                    conteo = len(df_dane[df_dane['Nombre Ocupación'] == nombre_dane])
                    
                    with st.container(border=True):
                        st.info(f"**Ocupación Equivalente en Colombia (IA):** {nombre_dane}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Afinidad de Significado", value=f"{similitud:.0%}")
                        with col2:
                            st.metric(label="Presencia en Encuesta DANE 2024", value=f"{conteo:,}".replace(',', '.'))
                        
                        with st.expander("Ver Descripción Completa del Perfil en Colombia (DANE)"):
                            st.write(descripcion_dane)
                else:
                    st.warning("No se encontró una equivalencia semántica directa en los datos de Colombia para esta profesión.")
                st.markdown("---")
        else:
            st.warning("Por favor, selecciona al menos una habilidad de la lista.")
else:
    st.error("La aplicación no pudo iniciar. Revisa los logs en 'Manage app' para más detalles si estás en la nube.")

.")