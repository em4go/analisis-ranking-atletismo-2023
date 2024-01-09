import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from library.functions import *

sidebar = st.sidebar
gender = sidebar.selectbox(
    "Select gender",
    ["Ambos", "Hombre", "Mujer"],
)
mixed_genders = gender == "Ambos"
df = load_data(gender)

sector = sidebar.selectbox("Select sector", ["Todos", *df["SECTOR"].unique()])

if sector != "Todos":
    df = df[df["SECTOR"] == sector]

event = sidebar.selectbox("Select event", ["Todos", *df["PRUEBA"].unique()])

skew = sidebar.radio(
    "Selecciona el sesgo de los gráficos", ["Ninguno", "Género", "Año"]
)

if event != "Todos":
    df = df[df["PRUEBA"] == event]
    df.sort_values(by=["MARCA_FLOAT"], inplace=True, ignore_index=True)
    st.dataframe(df)

    draw_charts(df, mixed_genders, skew)
else:
    st.write("Select an event on the sidebar to show the data")
