[README_streamlit.md](https://github.com/user-attachments/files/25677225/README_streamlit.md)
# Streamlit Data Apps

**MSc Business Analytics & Data Science · IE University · 2026**

A collection of interactive data applications built with Streamlit as part of coursework. Each app explores a different dataset and demonstrates data wrangling, visualization, and machine learning techniques.

---

## Apps

### ☕ Coffee Sales Dashboard — `coffee_app.py`

Interactive dashboard for analysing coffee sales performance across cities and products.

**Features:**
- Filter by city and product type
- KPIs: total units sold, total revenue, record count
- Revenue by product (bar chart)
- Sales trend over time (line chart)
- Temperature vs Iced Coffee sales scatter plot
- Raw data table

**Dataset:** `data/coffee_sales.csv`

---

### 🏠 Madrid Housing Price Predictor — `homework_app.py`

Multi-page app combining data exploration, interactive visualizations, a Random Forest ML model, and full model diagnostics for Madrid housing prices.

**Pages:**
| Page | Description |
|---|---|
| 📊 Data Explorer | Filter by neighborhood & property type, view summary statistics |
| 📈 Visualizations | Price vs area, median price by neighborhood, price distribution, correlation heatmap |
| 🤖 ML Predictor | Predict property price from user inputs via a trained Random Forest model |
| 📉 Model Diagnostics | Actual vs predicted plot, residuals plot, residual distribution |

**Model:** Random Forest Regressor · 100 estimators · R² reported on test set

**Dataset:** `data/housing_madrid.csv`

---

## Repo Structure

```
streamlit-apps/
├── coffee_app.py
├── homework_app.py
├── data/
│   ├── coffee_sales.csv
│   └── housing_madrid.csv
└── README.md
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/MarianGarabana/<repo-name>
cd <repo-name>
```

**2. Install dependencies**
```bash
pip install streamlit pandas plotly scikit-learn numpy
```

**3. Run an app**
```bash
# Coffee Sales Dashboard
streamlit run coffee_app.py

# Madrid Housing Price Predictor
streamlit run homework_app.py
```

> Make sure the CSV files are inside a `data/` folder at the root of the project.

---

## Stack

| Tool | Purpose |
|---|---|
| Streamlit | App framework & UI |
| Pandas | Data manipulation |
| Plotly Express | Interactive charts |
| Scikit-learn | Random Forest model, train/test split |
| NumPy | Numerical operations |
