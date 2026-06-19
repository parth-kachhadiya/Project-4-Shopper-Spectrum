# pages/1_EDA.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.helper import load_data

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.divider()

df = load_data()

# ── 1. TOP 10 COUNTRIES ───────────────────────────
st.subheader("🌍 Top 10 Countries by Orders")
top_countries = df.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False).head(10)
fig1 = px.bar(
    x=top_countries.values,
    y=top_countries.index,
    orientation='h',
    color=top_countries.values,
    color_continuous_scale='Viridis',
    labels={'x':'Number of Orders','y':'Country'},
    title='Top 10 Countries by Number of Orders'
)
fig1.update_layout(coloraxis_showscale=False, height=400)
st.plotly_chart(fig1, use_container_width=True)
st.divider()

# ── 2. TOP 10 PRODUCTS ────────────────────────────
st.subheader("🏆 Top 10 Best Selling Products")
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
fig2 = px.bar(
    x=top_products.values,
    y=top_products.index,
    orientation='h',
    color=top_products.values,
    color_continuous_scale='Magma',
    labels={'x':'Total Quantity Sold','y':'Product'},
)
fig2.update_layout(coloraxis_showscale=False, height=400)
st.plotly_chart(fig2, use_container_width=True)
st.divider()

# ── 3. MONTHLY REVENUE TREND ──────────────────────
st.subheader("📅 Monthly Revenue Trend")
monthly = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum().reset_index()
monthly['InvoiceDate'] = monthly['InvoiceDate'].astype(str)
fig3 = px.line(
    monthly, x='InvoiceDate', y='TotalPrice',
    markers=True,
    labels={'InvoiceDate':'Month','TotalPrice':'Total Revenue (£)'},
)
fig3.update_traces(line_color='steelblue', line_width=2.5)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)
st.divider()

# ── 4. QUARTERLY TREND ────────────────────────────
st.subheader("📆 Quarter-wise Revenue Trend")
quarterly = df.groupby('Quarter')['TotalPrice'].sum().reindex(['Q1','Q2','Q3','Q4']).reset_index()
fig4 = px.bar(
    quarterly, x='Quarter', y='TotalPrice',
    color='Quarter',
    color_discrete_map={'Q1':'#4E79A7','Q2':'#F28E2B','Q3':'#E15759','Q4':'#76B7B2'},
    labels={'TotalPrice':'Total Revenue (£)'},
    text_auto='.2s'
)
fig4.update_layout(showlegend=False, height=400)
st.plotly_chart(fig4, use_container_width=True)
st.divider()

# ── 5. CLOCK CHART ────────────────────────────────
st.subheader("🕐 Average Revenue by Hour (Clock View)")
hourly = df.groupby('Hour')['TotalPrice'].mean().reindex(range(24), fill_value=0)
angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
values = hourly.values
norm_values = (values - values.min()) / (values.max() - values.min()) * 0.4 + 0.55

fig5, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.fill(angles, norm_values, alpha=0.25, color='darkorange')
ax.plot(angles, norm_values, color='darkorange', linewidth=2.5)
ax.scatter(angles, norm_values, color='darkorange', s=60, zorder=5)
ax.set_xticks(angles)
ax.set_xticklabels([
    '12AM','1AM','2AM','3AM','4AM','5AM',
    '6AM','7AM','8AM','9AM','10AM','11AM',
    '12PM','1PM','2PM','3PM','4PM','5PM',
    '6PM','7PM','8PM','9PM','10PM','11PM'
], fontsize=8)
ax.set_yticklabels([])
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
peak_hour = hourly.idxmax()
ax.annotate(f'Peak {peak_hour}:00\n£{hourly[peak_hour]:.0f}',
            xy=(angles[peak_hour], norm_values[peak_hour]),
            xytext=(angles[peak_hour], norm_values[peak_hour]+0.15),
            fontsize=9, color='red', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.set_title('24H Revenue Clock', fontsize=13, fontweight='bold', pad=20)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.pyplot(fig5)
st.divider()

# ── 6. RFM DISTRIBUTIONS ──────────────────────────
st.subheader("📈 RFM Distributions")
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg(
    Recency  =('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo',   'nunique'),
    Monetary =('TotalPrice',  'sum')
).reset_index()

c1, c2, c3 = st.columns(3)
with c1:
    fig_r = px.histogram(rfm, x='Recency',   nbins=50, color_discrete_sequence=['steelblue'], title='Recency Distribution')
    st.plotly_chart(fig_r, use_container_width=True)
with c2:
    fig_f = px.histogram(rfm[rfm['Frequency']<=rfm['Frequency'].quantile(0.99)],
                         x='Frequency', nbins=50, color_discrete_sequence=['coral'], title='Frequency Distribution')
    st.plotly_chart(fig_f, use_container_width=True)
with c3:
    fig_m = px.histogram(rfm[rfm['Monetary']<=rfm['Monetary'].quantile(0.99)],
                         x='Monetary',  nbins=50, color_discrete_sequence=['mediumseagreen'], title='Monetary Distribution')
    st.plotly_chart(fig_m, use_container_width=True)
st.divider()

# ── 7. RFM CORRELATION HEATMAP ────────────────────
st.subheader("🔗 RFM Correlation Matrix")
fig7, ax7 = plt.subplots(figsize=(5,4))
sns.heatmap(rfm[['Recency','Frequency','Monetary']].corr(),
            annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, ax=ax7)
col1, col2, col3 = st.columns([1,1.5,1])
with col2:
    st.pyplot(fig7)