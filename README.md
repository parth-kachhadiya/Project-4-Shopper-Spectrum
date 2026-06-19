# 🛒 Shopper Spectrum
### Customer Segmentation & Product Recommendation System
> Domain: E-Commerce & Retail Analytics

---

## 📣 Problem Statement

E-commerce platforms generate massive transaction data daily. This project analyzes online retail transaction data to:
- Segment customers based on **RFM (Recency, Frequency, Monetary)** analysis
- Build a **Product Recommendation System** using Item-based Collaborative Filtering
- Deploy both as an interactive **Streamlit web application**

---

## 📁 Project Structure

```
shopper-spectrum/
├── app.py                        # Home page + navigation
├── ui_pages/
│   ├── 1_EDA.py                  # Exploratory Data Analysis
│   ├── 2_recommendation.py       # Product Recommender
│   └── 3_clustering.py           # Customer Segmentation
├── models/
│   ├── kmeans_model.pkl          # Trained KMeans model
│   ├── scaler.pkl                # StandardScaler
│   ├── label_map.pkl             # Cluster → Segment label map
│   ├── similarity_matrix.pkl     # Cosine similarity matrix
│   └── product_list.pkl          # All product names
├── utils/
│   └── helper.py                 # Shared loaders & functions
├── notebooks/
│   └── shopper_spectrum.ipynb    # Full analysis notebook
├── data/
│   └── online_retail.csv         # Raw dataset
├── requirements.txt
└── README.md
```

---

## 🧠 Approach

### Path A — Customer Segmentation

```
Raw Data → Clean → TotalPrice → RFM Table → Winsorize → Log Transform → Scale → KMeans K=4 → Labels
```

| Step | Detail |
|------|--------|
| Cleaning | Drop null CustomerID, cancelled invoices, Qty ≤ 0, Price ≤ 0 |
| RFM | Recency = days since last purchase, Frequency = unique invoices, Monetary = total spend |
| Outliers | Winsorize F & M at 1st–99th percentile |
| Skew fix | log1p transform on all 3 features |
| Scaling | StandardScaler |
| Model | KMeans K=4 (experimented KMeans, DBSCAN, Agglomerative) |

### Path B — Product Recommendation

```
Raw Data → Customer × Product Pivot → Transpose → Cosine Similarity Matrix → Top 5 Recommendations
```

---

## 🎯 Customer Segments

| Segment | Recency | Frequency | Monetary | Strategy |
|---------|---------|-----------|----------|----------|
| 🟢 High-Value | 12 days | 11.96 | £5,358 | Reward & retain — VIP programs |
| 🔵 Regular | 70 days | 4.08 | £1,698 | Upsell & cross-sell |
| 🟡 Occasional | 19 days | 2.06 | £524 | Re-engage with offers |
| 🔴 At-Risk | 183 days | 1.32 | £344 | Urgent win-back campaigns |

---

## 📊 EDA Highlights

- Top 10 countries by order volume
- Top 10 best-selling products
- Monthly revenue trend
- Quarter-wise revenue (Q1–Q4)
- 24-hour clock heatmap of purchase activity
- RFM distributions & correlation matrix

---

## 🔬 Model Experiments

| Model | Best K | Silhouette | Decision |
|-------|--------|------------|----------|
| KMeans | 2 | 0.4361 | ✅ Final choice (forced K=4) |
| Agglomerative | 2 | 0.4146 | Rejected — same as KMeans |
| DBSCAN | 2 | 0.2959 | Rejected — 15.8% noise at K=4 |

> Data naturally separates into 2 clusters. K=4 forced per business requirement. Silhouette=0.34 — acceptable.

---

## 📱 Streamlit App Pages

| Page | Features |
|------|----------|
| 🏠 Home | Project overview, dataset stats, RFM & segment explanation |
| 📊 EDA | 7 interactive charts (Plotly + Matplotlib) |
| 🛍️ Recommendation | Product search → 5 color-coded similar products |
| 🎯 Clustering | RFM input → segment prediction → business actions → 3D cluster graph with new point plotted |

---

## 🚀 Run Locally

**1. Clone & install dependencies**
```bash
https://github.com/parth-kachhadiya/Project-4-Shopper-Spectrum.git
cd Project-4-Shopper-Spectrum.git
pip install -r requirements.txt
```

**2. Run notebook first** (generates all model .pkl files)
```bash
jupyter notebook notebooks/model-design.ipynb
```

**3. Launch Streamlit**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
plotly
joblib
streamlit
jupyter
ipykernel
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | Pandas, NumPy, SciPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Model Persistence | Joblib |
| Web App | Streamlit |
| Notebook | Jupyter |

---

## 📌 Key Decisions

- **Why KMeans over DBSCAN?** DBSCAN produced 15.8% noise at K=4. KMeans more stable and interpretable for business labels.
- **Why log transform?** All 3 RFM features were right-skewed. log1p normalized distributions for better clustering.
- **Why Winsorize not drop?** Frequency (6.57%) and Monetary (9.80%) outlier % in 5–15% zone — dropping would lose too much data.
- **Why keep high Recency rows?** High recency = At-Risk customers. Dropping them removes an entire business segment.
- **Why item-based CF?** Simple, interpretable, no cold-start on products. Works well on retail transaction data.

---

*Shopper Spectrum — Built with 🛒 and Python*
