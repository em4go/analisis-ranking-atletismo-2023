import streamlit as st
import pandas as pd
import numpy as np
from library.functions import *
from wordcloud import WordCloud

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
    df.sort_values(by=["MARCA_FLOAT"], inplace=True, ignore_index=True)

event = sidebar.selectbox("Select event", ["Todos", *df["PRUEBA"].unique()])

if event != "Todos":
    df = df[df["PRUEBA"] == event]
    df.sort_values(by=["MARCA_FLOAT"], inplace=True, ignore_index=True)


def title(text, level=1):
    st.markdown(f"{'#'*level} {text}")


st.markdown(
    """
        # EDA App
        This is a simple EDA App built in Streamlit using the spanish athletics ranking data.
    """
)

st.write("## Exploratory Data Analysis")

dataset_statistics(df)

variables_statistics(df)

numerical_variables = df.dtypes == np.number
if numerical_variables.value_counts()[True] >= 2:
    interactions(df)

    correlations(df)

missing_values(df)

sample(df)
