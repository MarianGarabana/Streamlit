import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Housing Predictor", page_icon="🏠", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/housing_madrid.csv")
    return df

df = load_data()

@st.cache_resource
def train_model(df):
    feature_cols = ["area_sqm", "bedrooms", "bathrooms", "year_built",
                    "neighborhood", "property_type", "condition", "energy_rating", "has_parking"]
    X = df[feature_cols].copy()
    y = df["price_eur"]
    X = pd.get_dummies(X, columns=["neighborhood", "property_type", "condition", "energy_rating"],
                        drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    y_pred = model.predict(X_test) #need test set predictions for diagnostics
    return model, r2, X.columns.tolist(), y_test, y_pred

model, r2, model_columns, y_test, y_pred = train_model(df)  

st.title("🏠 Madrid Housing Price Predictor")

page = st.sidebar.radio("Navigate", ["📊 Data Explorer", "📈 Visualizations", "🤖 ML Predictor", "📉 Model Diagnostics"])
st.header(page)

if page == "📊 Data Explorer":
    selected_neighborhoods = st.sidebar.multiselect("Select Neighborhoods", df['neighborhood'].unique().tolist(),default=df['neighborhood'].unique().tolist())
    selected_property_types = st.sidebar.multiselect("Select Property Types",df['property_type'].unique().tolist(),default=df['property_type'].unique().tolist())
    filtered = df[df['neighborhood'].isin(selected_neighborhoods) & df['property_type'].isin(selected_property_types)]

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Properties", f"{filtered.shape[0]:,}")
    with col2:
        st.metric("Median Price", f"{filtered['price_eur'].median():,.2f}€")
    with col3:
        st.metric("Average Area (m²)", f"{filtered['area_sqm'].mean():,.2f}")

    st.dataframe(filtered, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(filtered.describe())

elif page == "📈 Visualizations":
    st.subheader("Price vs Area")
    color_option = st.selectbox("Color", ["neighborhood", "property_type", "condition", "energy_rating"])

    fig = px.scatter(
        df,
        x="area_sqm",
        y="price_eur",
        color=color_option,
        hover_data=["bedrooms", "neighborhood", "condition"],
        labels={"area_sqm": "Area (m²)", "price_eur": "Price (€)"},
        title="Area vs Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price by Neighborhood")

    median_prices = df.groupby("neighborhood")['price_eur'].median().reset_index()

    median_prices = median_prices.sort_values("price_eur", ascending=True)

    fig = px.bar(
        median_prices,
        y="neighborhood",
        x="price_eur",
        orientation="h",
        title="Median Price by Neighborhood",
        labels={"price_eur": "Median Price (€)", "neighborhood": "Neighborhood"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Distribution")

    fig = px.box(df, 
        x="neighborhood",
        y="price_eur",
        color="neighborhood",
        title="Price Distribution by Neighborhood",
        labels={"price_eur": "Price (€)", "neighborhood": "Neighborhood"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")

    numeric_cols = df[['area_sqm', 'bedrooms', 'bathrooms', 'year_built', 'price_eur', 'price_per_sqm']]

    corr_matrix = numeric_cols.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("The heatmap shows the correlation between the different features of the dataset. The darker the color, the stronger the correlation.")

elif page == "🤖 ML Predictor":

    st.metric("Model R² Score", f"{r2:.3f}")

    if r2 > 0.90:
        st.success("Excellent model performance!")
    elif r2 > 0.75:
        st.info("Good model performance.")
    else:
        st.warning("Model needs improvement.")

    st.subheader("Feature Importances")

    importance_df = pd.DataFrame({
            "Feature": model_columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(10)
        
    fig = px.bar(importance_df, 
                x="Importance", 
                y="Feature", 
                orientation="h",
                title="Top 10 Most Important Features")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predict a Property Price")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sqm)", min_value=25, max_value=300, value=100)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=1)
        year_built = st.number_input("Year Built", min_value=1960, max_value=2024, value=2000)

    with col2:
        neighborhood = st.selectbox("Neighborhood", df["neighborhood"].unique())
        property_type = st.selectbox("Property Type", df["property_type"].unique())
        condition = st.selectbox("Condition", df["condition"].unique())
        energy_rating = st.selectbox("Energy Rating", sorted(df["energy_rating"].unique()))
        has_parking = st.checkbox("Has Parking", value=True)

    if st.button("Predict Price"):
        input_dict = {
            "area_sqm": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "has_parking": has_parking,
            "neighborhood": neighborhood,
            "property_type": property_type,
            "condition": condition,
            "energy_rating": energy_rating
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, columns=["neighborhood", "property_type", "condition", "energy_rating"],
                                    drop_first=True)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        predicted_price = model.predict(input_encoded)[0]

        st.success(f"Estimated Price: **{predicted_price:,.0f}€**")
        st.metric("Price per sqm", f"{predicted_price/area:,.0f}€/sqm")

elif page == "📉 Model Diagnostics":

    # diagnostics dataframe
    diag_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Residual": y_test.values - y_pred
    })

    st.subheader("Actual vs Predicted Prices")
    fig1 = px.scatter(
        diag_df, x="Actual", y="Predicted",
        labels={"Actual": "Actual Price (€)", "Predicted": "Predicted Price (€)"},
        title="Actual vs Predicted"
    )

    # Add the ideal diagonal line
    min_val = diag_df[["Actual", "Predicted"]].min().min()
    max_val = diag_df[["Actual", "Predicted"]].max().max()
    fig1.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                   line=dict(color="red", dash="dash"))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Points close to the red line = accurate predictions")

    st.subheader("Residuals Plot")
    fig2 = px.scatter(
        diag_df, x="Predicted", y="Residual",
        labels={"Predicted": "Predicted Price (€)", "Residual": "Residual (€)"},
        title="Residuals should be randomly scattered around 0 (noise)"
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("A pattern here means the model has systematic bias.")

    st.subheader("Residual Distribution")
    fig3 = px.histogram(diag_df, x="Residual", nbins=30,
                        title="Distribution of Errors")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Ideally normally distributed (bell shape)")

