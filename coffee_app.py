import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Coffee Sales", layout="wide", page_icon="☕")

@st.cache_data
def load_data():
    df = pd.read_csv("data/coffee_sales.csv")
    return df

df = load_data()

st.title("☕ Coffee Sales Dashboard")

st.sidebar.header("Filters")

selected_city = st.sidebar.selectbox("City", ["All"] + df['city'].unique().tolist())

selected_products = st.sidebar.multiselect("Products", df["product"].unique().tolist(), default=df["product"].unique().tolist())

filtered = df[df["product"].isin(selected_products)]

if selected_city != "All":
    filtered = filtered[filtered["city"] == selected_city]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Units Sold", f"{filtered['units_sold'].sum():,}")

with col2:
    st.metric("Total Revenue", f"{filtered['revenue'].sum():,.2f}€")

with col3:
    st.metric("Number of Recors", f"{filtered['date'].count():,}")

st.subheader("Revenue by Product")

revenue_by_product = filtered.groupby("product")['revenue'].sum().reset_index()

fig = px.bar(
    revenue_by_product,
    x="product",
    y="revenue",
    color="product",
    title="Total Revenue by Product",
    labels={
        "product": "Product ",
        "revenue": "Revenue (€) ",
    }
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Sales Trend Over Time")

trend = filtered.groupby(['date', 'product'])['units_sold'].sum().reset_index()

fig = px.line(
    trend,
    x="date",
    y="units_sold",
    color="product",
    markers = True,
    title="Units Sold per Month"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Temperature vs Iced Coffee Sales")

iced = df[df['product'] == 'Iced Coffee']

fig = px.scatter(
    iced,
    x="temperature_c",
    y="units_sold",
    color= 'city',
    size="revenue",
    title="Do Hotter Days Sell More Iced Coffee?",
    labels = {"temperature_c": "Temperature (C)", "units_sold": "Units Sold"}
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Raw Data")
st.dataframe(filtered, use_container_width=True)
