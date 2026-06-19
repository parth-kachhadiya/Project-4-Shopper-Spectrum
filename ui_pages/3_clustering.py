# ui_pages/3_clustering.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats.mstats import winsorize
from utils.helper import load_models, load_data

st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")

# ── LOAD ──────────────────────────────────────────
kmeans, scaler, label_map, _, _ = load_models()
df = load_data()

# ── BUILD RFM + CLUSTER ───────────────────────────
@st.cache_data
def build_rfm():
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency  =('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo',   'nunique'),
        Monetary =('TotalPrice',  'sum')
    ).reset_index()
    rfm['Frequency'] = winsorize(rfm['Frequency'], limits=[0.01, 0.01])
    rfm['Monetary']  = winsorize(rfm['Monetary'],  limits=[0.01, 0.01])
    rfm['Recency_log']   = np.log1p(rfm['Recency'])
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log']  = np.log1p(rfm['Monetary'])
    return rfm

rfm = build_rfm()
rfm_scaled     = scaler.transform(rfm[['Recency_log','Frequency_log','Monetary_log']])
rfm['Cluster'] = kmeans.predict(rfm_scaled)
rfm['Segment'] = rfm['Cluster'].map(label_map)

# ── SEGMENT CONFIG ────────────────────────────────
config = {
    'High-Value': {
        'color'  : '#1e8c45',
        'bg'     : '#d4edda',
        'icon'   : '🟢',
        'desc'   : 'Recent, frequent & big spender. Highest priority customer. Reward with loyalty programs & exclusive offers.',
        'actions': ['Offer VIP membership', 'Early access to new products', 'Personalized thank-you rewards']
    },
    'Regular': {
        'color'  : '#1a6496',
        'bg'     : '#d0e9f7',
        'icon'   : '🔵',
        'desc'   : 'Steady buyer with moderate spend. Strong upsell & cross-sell potential.',
        'actions': ['Bundle product offers', 'Loyalty point incentives', 'Product upgrade suggestions']
    },
    'Occasional': {
        'color'  : '#856404',
        'bg'     : '#fff3cd',
        'icon'   : '🟡',
        'desc'   : 'Buys rarely with low spend. Needs re-engagement campaigns to increase visit frequency.',
        'actions': ['Limited time discount offers', 'Seasonal campaign targeting', 'Email re-engagement campaigns']
    },
    'At-Risk': {
        'color'  : '#721c24',
        'bg'     : '#f8d7da',
        'icon'   : '🔴',
        'desc'   : 'Inactive for long time. Urgent win-back strategy needed before permanently lost.',
        'actions': ['Win-back email with discount', 'Survey to understand churn reason', 'Special comeback offer']
    }
}

SEGMENT_COLORS = {
    'High-Value' : '#1e8c45',
    'Regular'    : '#1a6496',
    'Occasional' : '#856404',
    'At-Risk'    : '#721c24'
}

# ── HEADER ────────────────────────────────────────
st.title("🎯 Customer Segmentation")
st.markdown("Enter customer's **RFM values** to predict their segment.")
st.divider()

# ── INPUT ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    recency = st.number_input(
        "🕐 Recency (days since last purchase)",
        min_value=0, max_value=1000, value=30, step=1
    )
with col2:
    frequency = st.number_input(
        "🔁 Frequency (number of purchases)",
        min_value=1, max_value=500, value=5, step=1
    )
with col3:
    monetary = st.number_input(
        "💰 Monetary (total spend £)",
        min_value=0.0, max_value=1000000.0, value=500.0, step=10.0
    )

st.write("")
predict_btn = st.button("🔮 Predict Segment", use_container_width=True)
st.divider()

# ── PREDICTION + 3D ───────────────────────────────
if predict_btn:

    # predict
    inp       = np.log1p([[recency, frequency, monetary]])
    scaled    = scaler.transform(inp)
    cluster   = kmeans.predict(scaled)[0]
    segment   = label_map[cluster]
    cfg       = config[segment]

    # log values of new point for 3D
    new_r = np.log1p(recency)
    new_f = np.log1p(frequency)
    new_m = np.log1p(monetary)

    # ── SEGMENT BADGE ─────────────────────────────
    st.markdown(f"""
        <div style="
            background-color: {cfg['bg']};
            border-left: 6px solid {cfg['color']};
            padding: 20px 24px;
            border-radius: 10px;
            margin-bottom: 20px;
        ">
            <h2 style="color:{cfg['color']}; margin:0;">
                {cfg['icon']} {segment} Customer
            </h2>
            <p style="color:{cfg['color']}; margin-top:8px; font-size:15px;">
                {cfg['desc']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── INPUT SUMMARY ─────────────────────────────
    st.subheader("📋 Input Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("🕐 Recency",   f"{recency} days")
    c2.metric("🔁 Frequency", f"{frequency} orders")
    c3.metric("💰 Monetary",  f"£{monetary:,.2f}")

    # ── RECOMMENDED ACTIONS ───────────────────────
    st.write("")
    st.subheader("💡 Recommended Business Actions")
    for action in cfg['actions']:
        st.markdown(f"""
            <div style="
                background-color: {cfg['bg']};
                border-radius: 6px;
                padding: 10px 16px;
                margin-bottom: 8px;
                color: {cfg['color']};
                font-weight: 500;
            ">
                ✅ {action}
            </div>
        """, unsafe_allow_html=True)

    # ── 3D CLUSTER GRAPH + NEW POINT ──────────────
    st.divider()
    st.subheader("🌐 Your Position in Cluster Space")
    st.markdown(f"⭐ **Star marker** = your customer | Predicted: **{cfg['icon']} {segment}**")

    fig3d = px.scatter_3d(
        rfm,
        x='Recency_log',
        y='Frequency_log',
        z='Monetary_log',
        color='Segment',
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.4,
        labels={
            'Recency_log'  : 'Recency (log)',
            'Frequency_log': 'Frequency (log)',
            'Monetary_log' : 'Monetary (log)',
            'Segment'      : 'Segment'
        }
    )
    fig3d.update_traces(marker=dict(size=2), selector=dict(mode='markers'))

    # centroids
    centroids_log = scaler.inverse_transform(kmeans.cluster_centers_)
    fig3d.add_trace(go.Scatter3d(
        x=centroids_log[:, 0],
        y=centroids_log[:, 1],
        z=centroids_log[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='black', symbol='cross'),
        text=[label_map[i] for i in range(4)],
        textposition='top center',
        name='Centroids'
    ))

    # ── NEW POINT ─────────────────────────────────
    fig3d.add_trace(go.Scatter3d(
        x=[new_r],
        y=[new_f],
        z=[new_m],
        mode='markers+text',
        marker=dict(
            size=12,
            color=cfg['color'],
            symbol='diamond',
            line=dict(width=2, color='white')
        ),
        text=['⭐ You'],
        textposition='top center',
        name=f'Your Customer ({segment})'
    ))

    fig3d.update_layout(
        height=700,
        legend=dict(
            title='Segment',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='grey',
            borderwidth=1,
            font=dict(size=12)
        ),
        scene=dict(
            xaxis_title='Recency (log)',
            yaxis_title='Frequency (log)',
            zaxis_title='Monetary (log)'
        )
    )

    st.plotly_chart(fig3d, use_container_width=True)