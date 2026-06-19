# app.py
import streamlit as st
from utils.helper import load_data

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# ── NAVIGATION ────────────────────────────────────
pg = st.navigation([
    st.Page("ui_pages/1_EDA.py",            title="EDA",            icon="📊"),
    st.Page("ui_pages/2_recommendation.py", title="Recommendation", icon="🛍️"),
    st.Page("ui_pages/3_clustering.py",     title="Clustering",     icon="🎯"),
])
pg.run()

# ── HEADER ────────────────────────────────────────
df = load_data()

st.title("🛒 Shopper Spectrum")
st.markdown("#### Customer Segmentation & Product Recommendation Engine")
st.divider()

# ── DATASET STATS ─────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("Unique Customers",   f"{df['CustomerID'].nunique():,}")
c3.metric("Unique Products",    f"{df['Description'].nunique():,}")
c4.metric("Countries",          f"{df['Country'].nunique():,}")
st.divider()

# ── RFM EXPLANATION ───────────────────────────────
st.subheader("📊 What is RFM?")
r, f, m = st.columns(3)
with r:
    st.info("**🕐 Recency**\n\nDays since last purchase.\n\nLower = More Recent = Better")
with f:
    st.success("**🔁 Frequency**\n\nTotal number of purchases.\n\nHigher = More Loyal = Better")
with m:
    st.warning("**💰 Monetary**\n\nTotal amount spent.\n\nHigher = More Valuable = Better")
st.divider()

# ── SEGMENT GUIDE ─────────────────────────────────
st.subheader("🎯 Customer Segments")
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.success("**🟢 High-Value**\n\nRecent, frequent, big spenders.\n\nReward & retain.")
with s2:
    st.info("**🔵 Regular**\n\nSteady buyers.\n\nUpsell opportunities.")
with s3:
    st.warning("**🟡 Occasional**\n\nRare purchases.\n\nRe-engage with offers.")
with s4:
    st.error("**🔴 At-Risk**\n\nLong inactive.\n\nUrgent win-back needed.")