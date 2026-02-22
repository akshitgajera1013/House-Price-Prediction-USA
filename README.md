# 🇺🇸 USA Housing Market Intelligence Platform (Enterprise Edition)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Data%20Viz-Plotly-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An advanced, monolithic Python application built for institutional-grade real estate valuation and market telemetry. This system leverages a highly optimized **Decision Tree Regression Kernel** to process localized economic vectors and structural data, outputting highly accurate property valuations for the USA housing market.

## 🧠 System Architecture & Capabilities
This platform is divided into a 6-node monolithic architecture, processing raw demographic inputs into actionable financial intelligence.

### 1. Algorithmic Valuation Engine
Users input critical structural and economic topological data (Average Area Income, House Age, Total Rooms, Population Density). The Python-based UI processes these vectors through the Decision Tree inference kernel to generate an immediate market valuation.

### 2. Market Analytics & Radar Mapping
Generates a multi-dimensional radar topology of the target property, comparing its normalized vectors against simulated US national baselines. Includes a Gaussian probability density curve simulating local market asset distributions.

### 3. Hyperparameter Kernel Transparency
Complete transparency into the underlying Machine Learning architecture. The model is specifically tuned to prevent overfitting on the USA Housing dataset:
* **Criterion:** `friedman_mse` (Robust against extreme market outliers)
* **Max Depth:** `38` (Prevents infinite depth memorization)
* **Max Leaf Nodes:** `59` (Forces best-first growth mapping)
* **Min Samples Split / Leaf:** `39` / `15` (Ensures statistical significance per branch)

### 4. 15-Year ROI Financial Forecaster
Simulates multi-scenario compound annual growth rates (CAGR) for the predicted asset value. Trajectories are mapped for Bear (2%), Historical Average (4%), and Bull (7%) market conditions over a 15-year holding horizon.

### 5. Monte Carlo Volatility Simulation
Executes a 100-iteration random-walk mathematical simulation to model price volatility over a 12-month forward horizon. Automatically calculates maximum upside potential (95th Percentile) and Value at Risk (VaR).

### 6. Secure Data Export (JSON / CSV)
Generates an official Valuation Dossier tagged with a unique cryptographic Session ID. Enables base64-encoded, secure local downloads of the entire inference payload in both programmatic (JSON) and ledger (CSV) formats.

## 🛠️ Technical Stack
* **Core Logic & Computation:** `Python 3.x`, `numpy`
* **Data Processing & Pipelines:** `pandas`
* **Machine Learning Architecture:** `scikit-learn` (Decision Tree Regressor)
* **Interactive Data Visualization:** `plotly.express`, `plotly.graph_objects`
* **Frontend Delivery:** Custom Python-rendered UI engine with over 300 lines of injected, dynamic CSS (Glassmorphism, Keyframe Animations, Responsive Flexboxes).

## 📂 Repository Structure
├── app.py                  # Main monolithic Python application interface (1,000+ lines)
├── model.pkl               # Serialized Decision Tree Regressor model
├── scalar.pkl              # Serialized StandardScaler (if applicable)
├── encoder.pkl             # Serialized LabelEncoder (if applicable)
├── requirements.txt        # Python package dependencies
└── README.md               # System documentation

git clone https://github.com/akshitgajera1013/House-Price-Prediction-USA
cd usa-housing-intelligence
pip install -r requirements.txt
python -m streamlit run app.py
