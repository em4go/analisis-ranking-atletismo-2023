import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from random import randint
import math
from scipy import stats


def normalize_df(df):
    # Fix nullish dates
    df["F.N."] = df["F.N."].replace("-", np.nan)
    df["AÑO"] = df["AÑO"].replace("-", np.nan)

    # Fix data types
    df["F.N."] = pd.to_datetime(df["F.N."], format="%d/%m/%Y")
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%d/%m/%Y")
    df["ATLETA"] = df["ATLETA"].astype("str")
    df["AÑO"] = df["AÑO"].astype("float")
    df["CIUDAD"] = df["CIUDAD"].astype("str")

    df["ATLETA"] = df["ATLETA"].apply(str.title)

    return df


def draw_charts(df, mixed_genders=False, skew="Ninguno"):
    skew_dict = {"Ninguno": None, "Género": "SEXO", "Año": "AÑO"}
    opacity = 1
    color = skew_dict.get(skew, None)
    if color != None:
        opacity -= 0.3

    hist = px.histogram(
        df,
        x="MARCA_FLOAT",
        color=color,
        marginal="rug",
        hover_name="MARCA",
        hover_data=["ATLETA", "CLUB", "FECHA", "CIUDAD", "FED."],
        labels={
            "MARCA_FLOAT": "Mark in seconds (races), meters (jumps and throws) or points (decathlon)",
        },  # can specify one label per df column
        opacity=opacity,
        barmode="overlay",
    )
    st.markdown("## Histogram")
    st.plotly_chart(hist, use_container_width=True)

    violinmode = "group"  # or overlay

    violin = px.violin(
        df,
        x="MARCA_FLOAT",
        box=True,
        color=color,
        points="all",
        hover_name="MARCA",
        hover_data=["ATLETA", "CLUB", "FECHA", "CIUDAD", "FED."],
        violinmode=violinmode,
    )
    st.markdown("## Violin plot")
    st.plotly_chart(violin, use_container_width=True)

    st.markdown("## Number of marks by city")
    city_count = df["CIUDAD"].value_counts()
    city_plot = px.bar(city_count, x=city_count.index, y=city_count.values)
    st.plotly_chart(city_plot, use_container_width=True)

    st.markdown("## Number of marks by club")
    city_count = df["CLUB"].value_counts()
    club_plot = px.bar(city_count, x=city_count.index, y=city_count.values)
    st.plotly_chart(club_plot, use_container_width=True)


@st.cache_data
def load_data(gender="Ambos"):
    df = pd.read_parquet("sub18.parquet")
    if gender != "Ambos":
        df = df[df["SEXO"] == gender]

    df = normalize_df(df)
    return df


def simplify_bytes(n):
    n = int(n)
    for b in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {b}"
        n /= 1024


def get_category(df, variable_name):
    variable = df[variable_name]
    i = randint(0, len(variable) - 1)
    value = variable[i]
    value_type = type(value)
    if "float" in str(value_type):
        return "Real number"
    elif "int" in str(value_type):
        return "Integer"
    elif "str" in str(value_type):
        if len(variable.unique()) <= 30:
            return "Categorical"
        return "Text"
    elif "bool" in str(value_type):
        return "Boolean"
    elif "datetime" in str(value_type):
        return "Date"
    elif "time" in str(value_type):
        return "Date"
    elif "timedelta" in str(value_type):
        return "Date"
    else:
        return value_type


def dataset_statistics(df):
    # Number of variables
    number_of_variables = len(df.columns)
    # Number of observations
    number_of_observations = len(df)
    # Missing cells
    missing_cells = df.isnull().sum().sum()
    # Missing cells (%)
    missing_cells_p = round(
        missing_cells / (number_of_variables * number_of_observations), 3
    )
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    # Duplicate rows (%)
    duplicate_rows_p = duplicate_rows / number_of_observations
    # Total size in memory
    total_size = df.memory_usage(deep=True).sum()
    # Average record size in memory
    average_record_size = total_size / number_of_observations

    # Variable types
    variable_types = df.dtypes.value_counts().to_frame()
    types_count = {}
    for col in df.columns:
        type_name = get_category(df, col)
        if type_name in types_count:
            types_count[type_name] += 1
        else:
            types_count[type_name] = 1

    col1, col2 = st.columns(2)

    statistics_table = f"""
    | Statistic | Value |
    | --- | --- |
    | Number of variables | {number_of_variables} |
    | Number of observations | {number_of_observations} |
    | Missing cells | {missing_cells} |
    | Missing cells (%) | {missing_cells_p} |
    | Duplicate rows | {duplicate_rows} |
    | Duplicate rows (%) | {duplicate_rows_p} |
    | Total size in memory | {simplify_bytes(total_size)} |
    | Average record size in memory | {simplify_bytes(average_record_size)} |
    """

    variable_types_table = f"""
    | Variable type | Count |
    | --- | --- |
    """
    for key, value in types_count.items():
        variable_types_table += f"""| {key} | {value} |
    """

    col1.markdown("### Dataset statistics")
    col1.markdown(statistics_table)
    col2.markdown("### Variable types")
    col2.markdown(variable_types_table)


def variables_statistics(df):
    chart = None
    wordcloud = None
    more_details = None
    more_details_text = ""
    st.markdown("## Variables")

    # Select variable
    variable_name = st.selectbox("Select variable", df.columns)
    variable = df[variable_name]

    # Variable type
    variable_type = get_category(df, variable_name)

    # Common to all types
    number_of_observations = len(variable)
    distinct = variable.nunique()
    distinct_p = round(distinct / number_of_observations * 100, 4)
    missing = variable.isnull().sum()
    missing_p = round(missing / number_of_observations * 100, 4)
    memory_size = variable.memory_usage(deep=True)

    st.write(f"### {variable_name}")
    add_to_text = ""
    if variable_type == "Real number":
        add_to_text = " $(\mathbb{R})$"
    if variable_type == "Integer":
        add_to_text = " $(\mathbb{Z})$"
    st.write(f"#### {variable_type} {add_to_text}")
    col1, col2 = st.columns(2)
    stats_table = f"""
            | Statistic | Value |
            | --- | --- |
            | Variable type | {variable_type} |
            | Number of observations | {number_of_observations} |
            | Distinct | {distinct} |
            | Distinct (%) | {distinct_p} |
            | Missing | {missing} |
            | Missing (%) | {missing_p} |
            | Memory size | {simplify_bytes(memory_size)} |"""

    if variable_type == "Real number" or variable_type == "Integer" and distinct > 30:
        infinites = variable.isin([np.inf, -np.inf]).sum()
        infinites_p = round(infinites / number_of_observations * 100, 4)
        mean = round(variable.mean(), 4)

        minimum = variable.min()
        maximum = variable.max()
        zeros = variable.isin([0]).sum()
        zeros_p = round(zeros / number_of_observations * 100, 4)
        negative = variable[variable < 0]
        number_of_negative = len(negative)
        negative_p = round(number_of_negative / number_of_observations * 100, 4)

        # More details
        # # Quantile statistics
        fifth_percentile = round(variable.quantile(0.05), 4)
        q1 = round(variable.quantile(0.25), 4)
        median = variable.median()
        q3 = round(variable.quantile(0.75), 4)
        ninety_fifth_percentile = round(variable.quantile(0.95), 4)
        var_range = maximum - minimum
        iqr = round(q3 - q1, 4)

        # # Descriptive statistics
        standard_deviation = round(variable.std(), 4)
        variance = round(variable.var(), 4)
        coefficient_of_variation = round(standard_deviation / mean, 4)
        kurtosis = round(variable.kurtosis(), 4)
        # mean
        median_absolute_variation = round(stats.median_abs_deviation(variable), 4)
        skewness = round(variable.skew(), 4)
        var_sum = variable.sum()
        monotonicity = (
            variable.is_monotonic_increasing and variable.is_monotonic_decreasing
        )
        monotonicity_text = "Monotonic" if monotonicity else "Not monotonic"

        stats_table += f"""
            | Infinite | {infinites} |
            | Infinite (%) | {infinites_p} |
            | Mean | {mean} |
            | Minimum | {minimum} |
            | Maximum | {maximum} |
            | Zeros | {zeros} |
            | Zeros (%) | {zeros_p} |
            | Negative | {number_of_negative} |
            | Negative (%) | {negative_p} |
        """

        # Histograma de la distribución de la variable
        chart = px.histogram(
            df,
            x=variable_name,
            nbins=round(math.sqrt(number_of_observations)),
            marginal="rug",
        )

        more_details = st.expander("More details")
        quantile_statistics_table = f"""
            | Statistic | Value |
            | --- | --- |
            | Minimum | {minimum} |
            | 5th percentile | {fifth_percentile} |
            | 25th percentile | {q1} |
            | Median | {median} |
            | 75th percentile | {q3} |
            | 95th percentile | {ninety_fifth_percentile} |
            | Maximum | {ninety_fifth_percentile} |
            | Range | {maximum} |
            | IQR | {iqr} |
        """
        descriptive_statistics_table = f"""
            | Statistic | Value |
            | --- | --- |
            | Standard deviation | {standard_deviation} |
            | Variance | {variance} |
            | Coefficient of variation | {coefficient_of_variation} |
            | Kurtosis | {kurtosis} |
            | Mean | {mean} |
            | Median absolute variation | {median_absolute_variation} |
            | Skewness | {skewness} |
            | Sum | {var_sum} |
            | Monotonicity | {monotonicity_text} |
            """
    if variable_type == "Text":
        text = " ".join(variable)
        text = variable.value_counts().to_dict()
        img = "runner.jpg"
        mask = generate_mask(img)
        wordcloud = wordcloud_from_dict(text, mask).to_array()

    if (
        variable_type == "Categorical"
        or variable_type == "Boolean"
        or (variable_type == "Integer" and distinct <= 30)
    ):
        unique_values = variable.value_counts()
        chart = px.bar(unique_values, x=unique_values.index, y=unique_values.values)

    with col1:
        st.markdown(stats_table)
    if chart is not None:
        with col2:
            st.plotly_chart(chart, use_container_width=True)
    if wordcloud is not None:
        with col2:
            st.image(wordcloud)
    if more_details is not None:
        with more_details:
            md_col1, md_col2 = st.columns(2)
            md_col1.markdown("### Quantile statistics")
            md_col1.markdown(quantile_statistics_table)
            md_col2.markdown("### Descriptive statistics")
            md_col2.markdown(descriptive_statistics_table)


def sample(df):
    st.write("## Sample")
    sample_size = st.slider("Sample size", 5, 20)
    first, last = st.tabs(["First", "Last"])
    with first:
        st.dataframe(df.head(sample_size))
    with last:
        st.dataframe(df.tail(sample_size))


def interactions(df):
    st.write("## Interactions")
    numeric_variables = list(df.select_dtypes(include=[np.number]).columns)
    x_variable = st.selectbox("Select x variable", numeric_variables)
    y_variable = st.selectbox("Select y variable", numeric_variables)
    fig = px.scatter(
        df,
        x=x_variable,
        y=y_variable,
        hover_name=df["MARCA"],
        hover_data=["ATLETA", "PRUEBA", "CLUB", "CIUDAD", "FED."],
        trendline="ols",
    )
    st.plotly_chart(fig, use_container_width=True)


def correlations(df):
    st.write("## Correlations")
    numerical = df.select_dtypes(include=[np.number])
    pearson_corr = numerical.corr(method="pearson")
    heatmap, table = st.tabs(["Heatmap", "Table"])
    with heatmap:
        fig = px.imshow(pearson_corr, zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    with table:
        st.table(pearson_corr)


def missing_values(df):
    st.write("## Missing values")
    bar_tab, nullity_matrix_tab, bar_tab_p = st.tabs(
        ["Bar chart", "Nullity matrix", "Bar chart (%)"]
    )
    # missing_count = df.isnull().sum()
    # missing_p = round(missing / len(df) * 100, 4)
    missing = df.isnull()
    missing_sum = missing.sum()
    missing_sum_p = round(missing_sum / len(df) * 100, 4)

    bar_chart = px.bar(missing_sum, x=missing.columns, y=missing_sum.values, height=500)
    bar_tab.plotly_chart(bar_chart, use_container_width=True)

    nullity_matrix = px.imshow(missing, color_continuous_scale="thermal")
    nullity_matrix_tab.plotly_chart(nullity_matrix, use_container_width=True)

    bar_chart_p = px.bar(
        missing_sum_p,
        x=missing.columns,
        y=missing_sum_p.values,
        height=500,
    )
    bar_chart_p.update_yaxes(range=[0, 100])
    bar_tab_p.plotly_chart(bar_chart_p, use_container_width=True)


# A function to generate the mask for the cloud
def generate_mask(img):
    mask = np.array(Image.open(img))
    return mask


# A function to generate the word cloud from text
def gen_wordcloud(mask=None):
    cloud = WordCloud(
        scale=3,
        max_words=150,
        colormap="autumn",
        mask=mask,
        background_color="#0e1117",
        collocations=True,
        contour_color="#EEEEEE",
        contour_width=5,
        stopwords=STOPWORDS,
    )
    return cloud


def wordcloud_from_dict(data, mask=None):
    cloud = gen_wordcloud(mask).generate_from_frequencies(data)
    return cloud


def wordcloud_from_text(data, mask=None):
    cloud = gen_wordcloud(mask).generate_from_text(data)
    return cloud
