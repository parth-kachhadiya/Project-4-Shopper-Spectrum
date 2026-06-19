# utils.py
import joblib
import pandas as pd
import streamlit as st


MODEL_SOURCE_DIR = "models"
DATA_SOURCE_DIR = "data"

@st.cache_resource
def load_models():
    kmeans     = joblib.load(f'{MODEL_SOURCE_DIR}/kmeans_model.pkl')
    scaler     = joblib.load(f'{MODEL_SOURCE_DIR}/scaler.pkl')
    label_map  = joblib.load(f'{MODEL_SOURCE_DIR}/label_map.pkl')
    sim_matrix = joblib.load(f'{MODEL_SOURCE_DIR}/similarity_matrix.pkl')
    products   = joblib.load(f'{MODEL_SOURCE_DIR}/product_list.pkl')
    return kmeans, scaler, label_map, sim_matrix, products

@st.cache_data
def load_data():
    df = pd.read_csv(f'{DATA_SOURCE_DIR}/online_retail.csv', encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID','Description'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df.drop_duplicates(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID']  = df['CustomerID'].astype(int)
    df['TotalPrice']  = df['Quantity'] * df['UnitPrice']
    df['Month']       = df['InvoiceDate'].dt.to_period('M')
    df['Quarter']     = df['InvoiceDate'].dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'})
    df['Hour']        = df['InvoiceDate'].dt.hour
    return df

def predict_segment(recency, frequency, monetary, kmeans, scaler, label_map):
    import numpy as np
    inp = np.log1p([[recency, frequency, monetary]])
    scaled = scaler.transform(inp)
    cluster = kmeans.predict(scaled)[0]
    return label_map[cluster]

def recommend_products(product_name, sim_matrix, n=5):
    if product_name not in sim_matrix.index:
        return []
    similar = sim_matrix[product_name].sort_values(ascending=False)
    similar = similar.drop(product_name)
    top = similar.head(n)
    return [(prod, round(score, 4)) for prod, score in top.items()]