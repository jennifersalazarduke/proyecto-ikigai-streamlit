# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib # Para cargar tu modelo k-NN
from pathlib import Path
# app.py (versión final y mejorada)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Configuración de la Página (Título, Ícono, Layout) ---
st.set_page_config(
    page_title="Ikigai-ML: Tu Orientador Profesional",
    page_icon="🤖",
    layout="wide"
)

# --- Carga de Datos y Modelos (con caché para alta velocidad) ---
@st.cache_data
def cargar_activos():
    data_path = Path("./data")
    try:
        # Cargar el recomendador O*NET
        knn_model = joblib.load(data_path / "knn_model.pkl")
        pivot_onet = pd.read_parquet(data_path / "mat_full.parquet")
        # Cargar un DataFrame simple solo con los títulos de O*NET para mostrar
        df_onet_titulos = pd.read_parquet(data_path / "onet_titles.parquet") # Necesitarás guardar este archivo
        
        # Cargar el puente inteligente
        df_puente = pd.read_parquet(data_path / "puente_onet_dane_ia.parquet")
        
        # Cargar los datos del DANE para el conteo
        df_dane = pd.read_parquet(data_path / "dane_enriquecido_final_2024.parquet")
    except FileNotFoundError as e:
        st.error(f"Error al cargar un archivo necesario: {e}. Asegúrate de que todos los archivos estén en la carpeta 'data'.")
        return None, None, None, None, None

    return knn_model, pivot_onet, df_onet_titulos, df_puente, df_dane

# Cargamos los activos
knn_model, pivot_onet, df_onet_titulos, df_puente, df_dane = cargar_activos()

# --- Preparación de la lista de habilidades para el widget ---
if pivot_onet is not None:
    # Creamos la lista de habilidades a partir de las columnas de la matriz O*NET
    lista_completa_habilidades = sorted([col for col in pivot_onet.columns if col not in ['Title', 'Description']])
else:
    lista_completa_habilidades = ["Error al cargar habilidades..."]

# Placeholder para tu función de vectorización
def vector_usuario(user_skills, reference_matrix):
    # ... Pega aquí tu función vector_usuario REAL ...
    # Esta función debe tomar la lista de skills y devolver el vector numpy
    vector = np.zeros((1, len(reference_matrix.columns)))
    present_skills = [skill for skill in user_skills if skill in reference_matrix.columns]
    if present_skills:
        for skill in present_skills:
            vector[0, reference_matrix.columns.get_loc(skill)] = 1
        return vector / vector.sum()
    return vector


# --- Interfaz de Usuario de la Aplicación ---
st.title("🤖 Proyecto Ikigai-ML")
st.header("Tu Orientador Profesional con IA")
st.write("Descubre qué profesiones se ajustan a tus habilidades y conoce su contexto en el mercado laboral colombiano.")

# --- NUEVO WIDGET DE SELECCIÓN MÚLTIPLE ---
user_skills_seleccionadas = st.multiselect(
    "Selecciona tus habilidades de la lista (puedes escribir para buscar):",
    options=lista_completa_habilidades,
    placeholder="Elige una o varias habilidades"
)

# Botón para ejecutar
if st.button("Encontrar mi Ikigai ✨"):
    if user_skills_seleccionadas and knn_model is not None:
        
        # --- 1. Generar Recomendaciones de O*NET ---
        st.subheader("Tus Profesiones Recomendadas (Basado en O*NET)")

        u_vec = vector_usuario(user_skills_seleccionadas, pivot_onet)
        
        # Usamos el modelo k-NN para encontrar las 5 profesiones más cercanas
        distances, indices = knn_model.kneighbors(u_vec, n_neighbors=5)
        
        # Obtenemos los títulos de las profesiones recomendadas
        onet_results_titulos = df_onet_titulos.iloc[indices[0]]

        # --- 2. Enriquecer y Mostrar los Resultados ---
        for index, row in onet_results_titulos.iterrows():
            titulo_onet = row['Title']
            
            st.markdown(f"#### {titulo_onet}")
            
            # Buscamos la información local en nuestro puente inteligente
            info_local = df_puente[df_puente['Onet_Title'] == titulo_onet]
            
            if not info_local.empty:
                nombre_dane = info_local['Dane_Name'].iloc[0]
                descripcion_dane = info_local['Dane_Description'].iloc[0]
                similitud = info_local['Similarity_Score'].iloc[0]
                
                # Buscamos el conteo en los datos del DANE
                conteo = len(df_dane[df_dane['Nombre Ocupación'] == nombre_dane])
                
                with st.expander(f"Ver contexto para '{nombre_dane}' en Colombia 🇨🇴"):
                    st.info(f"**Similitud de Significado con la profesión de O*NET:** {similitud:.2f} (de 0 a 1)")
                    st.metric(label="Presencia en Encuesta Nacional 2024", value=f"{conteo:,}".replace(',', '.'))
                    st.markdown(f"**Descripción del Perfil (DANE):** {descripcion_dane}")
            else:
                st.warning("No se encontró una equivalencia semántica directa en los datos de Colombia para esta profesión.")
    else:
        st.warning("Por favor, selecciona al menos una habilidad de la lista.")