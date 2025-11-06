
import streamlit as st
import pandas as pd
import numpy as np
import pickle, json

st.set_page_config(page_title="bike-ai-dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("bike_ai_dataset.csv")

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

st.title("🏍️ bike-ai-dashboard")
st.caption("Prediksi Days_to_Sell, EDA, dan Profit & Margin (Bahasa Indonesia).")

# Overview
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Record", len(df))
with c2: st.metric("Total Fitur", df.shape[1])
with c3: st.metric("Rata-rata Days_to_Sell", f"{df['Days_to_Sell'].mean():.2f}")
with c4: st.metric("Median Days_to_Sell", f"{df['Days_to_Sell'].median():.2f}")

st.markdown("---")

# Sidebar
st.sidebar.header("Filter")
cat_cols = [c for c in df.columns if df[c].dtype != float and c != "Days_to_Sell"]
df_filt = df.copy()
if cat_cols:
    cat_col = st.sidebar.selectbox("Kolom kategorikal", options=cat_cols)
    vals = sorted(df[cat_col].dropna().unique().tolist())
    sel_vals = st.sidebar.multiselect("Nilai", options=vals, default=vals[:5])
    if sel_vals:
        df_filt = df_filt[df_filt[cat_col].isin(sel_vals)]

num_cols = [c for c in df.columns if df[c].dtype == float and c != "Days_to_Sell"]
num_col = st.sidebar.selectbox("Kolom numerik untuk distribusi", options=num_cols) if num_cols else None

# EDA
cA, cB = st.columns(2)
with cA:
    st.subheader("Distribusi Days_to_Sell (filtered)")
    st.bar_chart(df_filt["Days_to_Sell"].value_counts().sort_index())
with cB:
    st.subheader(f"Distribusi {num_col if num_col else '(tidak ada)'}")
    if num_col:
        st.line_chart(df_filt[num_col])
    else:
        st.write("Tidak ada kolom numerik selain target.")

st.markdown("---")

# Prediction
st.header("🤖 Prediksi Days_to_Sell")
row_index = st.number_input("Index baris untuk prediksi", min_value=0, max_value=len(df)-1, value=0, step=1)
row = df.iloc[[row_index]].copy()

key_features = [f for f in ["Aging  Sales", "Ending Inventory", "Unit beli", "Biaya Rekondisi"] if f in df.columns]
if key_features:
    st.subheader("What-if Simulator (ubah fitur utama)")
    for k in key_features:
        min_v, max_v = float(df[k].min()), float(df[k].max())
        default = float(row.iloc[0][k])
        row.loc[row.index, k] = st.slider(k, min_value=min_v, max_value=max_v, value=default)

X_cols = [c for c in feature_names if c in row.columns]
pred = model.predict(row[X_cols])[0]
st.success(f"Prediksi Days_to_Sell: **{pred:.3f}**")

# Profit & Margin
st.markdown("---")
st.header("💰 Profit & Margin")

st.write("Pilih kolom yang sesuai untuk perhitungan profit & margin.")

num_cols_all = [c for c in df.columns if df[c].dtype in [float, int]]

col1, col2, col3, col4 = st.columns(4)
with col1:
    col_sell = st.selectbox("Kolom Harga Jual (Selling Price)", options=["(tidak ada)"] + num_cols_all, index=0)
with col2:
    default_buy_idx = (num_cols_all.index("Unit beli")+1) if "Unit beli" in num_cols_all else 0
    col_buy = st.selectbox("Kolom Harga Beli / Cost", options=["(tidak ada)"] + num_cols_all, index=default_buy_idx)
with col3:
    default_recon_idx = (num_cols_all.index("Biaya Rekondisi")+1) if "Biaya Rekondisi" in num_cols_all else 0
    col_recon = st.selectbox("Kolom Biaya Rekondisi (opsional)", options=["(tidak ada)"] + num_cols_all, index=default_recon_idx)
with col4:
    col_other = st.selectbox("Kolom Other Revenue (opsional)", options=["(tidak ada)"] + num_cols_all, index=0)

df_profit = df_filt.copy()

if col_sell != "(tidak ada)" and col_buy != "(tidak ada)":
    sell = df_profit[col_sell].astype(float)
    buy = df_profit[col_buy].astype(float)
    recon = df_profit[col_recon].astype(float) if col_recon != "(tidak ada)" else 0.0
    other_rev = df_profit[col_other].astype(float) if col_other != "(tidak ada)" else 0.0

    df_profit["Profit"] = sell - (buy + recon) + other_rev
    denom = (buy + recon).replace(0, np.nan)
    df_profit["Margin_%"] = (df_profit["Profit"] / denom) * 100
    df_profit["Margin_%"] = df_profit["Margin_%"].fillna(0)

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Total Profit (filtered)", f"{df_profit['Profit'].sum():,.0f}")
    with m2: st.metric("Avg Profit per Unit", f"{df_profit['Profit'].mean():,.0f}")
    with m3: st.metric("Avg Margin %", f"{df_profit['Margin_%'].mean():.2f}%")

    dim_options = [c for c in ["Category", "Lokasi Aktual", "type sold"] if c in df_profit.columns]
    dim = st.selectbox("Breakdown berdasarkan", options=dim_options if dim_options else ["(tidak tersedia)"])
    if dim != "(tidak tersedia)":
        agg = df_profit.groupby(dim).agg({"Profit":"sum", "Margin_%":"mean", "Days_to_Sell":"mean"}).sort_values("Profit", ascending=False)
        st.subheader("Performa per Grup")
        st.dataframe(agg, use_container_width=True)
        st.bar_chart(agg["Profit"])

        st.caption("Tip: Nilai dimensi adalah kode hasil encoding. Gunakan `categorical_mappings.json` untuk melihat label aslinya saat ingin memperkaya UI.")
else:
    st.info("Pilih setidaknya kolom **Harga Jual** dan **Harga Beli** untuk menghitung profit & margin.")

st.caption("© 2025 bike-ai-dashboard")
