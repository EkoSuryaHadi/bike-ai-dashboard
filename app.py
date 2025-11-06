
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json

st.set_page_config(page_title="Bike AI Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("bike_ai_dataset.csv")
    return df

@st.cache_resource
def load_model():
    with open("model_days_to_sell.pkl", "rb") as f:
        pack = pickle.load(f)
    return pack["model"], pack["feature_names"]

@st.cache_data
def load_mappings():
    with open("categorical_mappings.json", "r") as f:
        return json.load(f)

df = load_data()
model, feature_names = load_model()
mappings = load_mappings()

st.title("🏍️ Bike AI Dashboard — Streamlit")
st.caption("Prediksi waktu penjualan (Days_to_Sell) + EDA interaktif")

# --- Overview cards
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Record", len(df))
with c2: st.metric("Total Fitur", df.shape[1])
with c3: st.metric("Rata-rata Days_to_Sell", f"{df['Days_to_Sell'].mean():.2f}")
with c4: st.metric("Median Days_to_Sell", f"{df['Days_to_Sell'].median():.2f}")

st.markdown("---")

# --- Sidebar Filters
st.sidebar.header("Filters")
cat_cols = [c for c in df.columns if df[c].dtype != float and c != "Days_to_Sell"]
if cat_cols:
    cat_col = st.sidebar.selectbox("Filter Kolom Kategorikal", options=cat_cols)
    vals = sorted(df[cat_col].dropna().unique().tolist())
    sel_vals = st.sidebar.multiselect("Nilai", options=vals, default=vals[:5])
    df_filt = df[df[cat_col].isin(sel_vals)] if sel_vals else df.copy()
else:
    st.sidebar.info("Tidak ada kolom kategorikal terdeteksi.")
    df_filt = df.copy()

num_cols = [c for c in df.columns if df[c].dtype == float and c != "Days_to_Sell"]
if num_cols:
    num_col = st.sidebar.selectbox("Kolom Numerik untuk Distribusi", options=num_cols)
else:
    num_col = None

# --- EDA charts
cA, cB = st.columns(2)
with cA:
    st.subheader("Distribusi Days_to_Sell (Filtered)")
    st.bar_chart(df_filt["Days_to_Sell"].value_counts().sort_index())
with cB:
    st.subheader(f"Distribusi {num_col if num_col else '(tidak ada)'}")
    if num_col:
        st.line_chart(df_filt[num_col])
    else:
        st.write("Tidak ada kolom numerik lain.")

st.markdown("---")

# --- Prediction Panel
st.header("🤖 Prediksi Days_to_Sell")
st.write("Pilih baris data untuk diprediksi atau lakukan simulasi 'what-if'.")

# Select a row
row_index = st.number_input("Index baris untuk prediksi", min_value=0, max_value=len(df)-1, value=0, step=1)
row = df.iloc[[row_index]].copy()

# What-if controls for beberapa fitur penting (jika ada)
key_features = [f for f in ["Aging  Sales", "Ending Inventory", "Unit beli", "Biaya Rekondisi"] if f in df.columns]
if key_features:
    st.subheader("What-if Simulator (ubah fitur utama)")
    for k in key_features:
        min_v, max_v = float(df[k].min()), float(df[k].max())
        default = float(row.iloc[0][k])
        row.loc[row.index, k] = st.slider(k, min_value=min_v, max_value=max_v, value=default)

# Align features to model
X_cols = [c for c in feature_names if c in row.columns]
X = row[X_cols]

pred = model.predict(X)[0]
st.success(f"Prediksi Days_to_Sell: **{pred:.3f}**")

st.markdown("---")

# --- Feature Importance (if available)
try:
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(15)
    st.subheader("Fitur Paling Berpengaruh")
    st.dataframe(imp_df, use_container_width=True)
except Exception:
    st.info("Model tidak mendukung feature_importances_.")

st.caption("© 2025 Bike AI Dashboard — Streamlit")
