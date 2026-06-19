# ui_pages/2_recommendation.py
import streamlit as st
from utils.helper import load_models

st.set_page_config(page_title="Product Recommendation", page_icon="🛍️", layout="wide")

# ── LOAD MODELS ───────────────────────────────────
_, _, _, sim_matrix, products = load_models()

# ── HEADER ────────────────────────────────────────
st.title("🛍️ Product Recommender")
st.markdown("Enter a product name to get **5 similar product recommendations** based on customer purchase patterns.")
st.divider()

# ── INPUT ─────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    selected_product = st.selectbox(
        "🔍 Select or Search Product Name",
        options=sorted(products),
        index=None,
        placeholder="Type to search product..."
    )

with col2:
    st.write("")
    st.write("")
    recommend_btn = st.button("🔎 Get Recommendations", use_container_width=True)

st.divider()

# ── RESULTS ───────────────────────────────────────
if recommend_btn:
    if not selected_product:
        st.warning("Please select a product first.")
    else:
        similar = sim_matrix[selected_product].sort_values(ascending=False)
        similar = similar.drop(selected_product).head(5)
        results = [(prod, round(score, 4)) for prod, score in similar.items()]

        st.subheader(f"🎯 Top 5 Recommendations for:")
        st.markdown(f"### `{selected_product}`")
        st.write("")

        for i, (product, score) in enumerate(results, 1):

            # ── COLOR BY SCORE ────────────────────
            if score >= 0.7:
                color    = "#1e8c45"
                bg       = "#d4edda"
                badge    = "🟢 Strong Match"
            elif score >= 0.5:
                color    = "#856404"
                bg       = "#fff3cd"
                badge    = "🟡 Moderate Match"
            else:
                color    = "#721c24"
                bg       = "#f8d7da"
                badge    = "🔴 Weak Match"

            st.markdown(f"""
                <div style="
                    background-color: {bg};
                    border-left: 5px solid {color};
                    padding: 12px 18px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                ">
                    <span style="font-size:16px; font-weight:600; color:{color};">
                        {i}. {product}
                    </span>
                    <span style="float:right; font-size:13px; color:{color};">
                        {badge} &nbsp;|&nbsp; Score: {score}
                    </span>
                </div>
            """, unsafe_allow_html=True)