# Sales Forecasting & Demand Intelligence System 📈

An end-to-end Machine Learning pipeline and analytical dashboard built on **4 years of Superstore retail transaction data (2015–2018)**. The application provides demand intelligence, detects anomalies, segments products, and generates production-ready time-series forecasts using **Facebook Prophet, SARIMA, and XGBoost** models.

🔗 **Live Deployment:** [salesforecastingajaikumar-rq5hlbvnjr38ac2xyfj3i7.streamlit.app](https://salesforecastingajaikumar-rq5hlbvnjr38ac2xyfj3i7.streamlit.app/)

---

## 🚀 Key Dashboard Pages

1. **🏠 Home & KPI Dashboard:** A high-level overview of retail operations showing total sales ($2.29M), top regions, fastest-growing departments, and a summary of historical sales trends.
2. **🔮 Forecast Explorer:** Interactive demand outlook comparing **Prophet, SARIMA, and XGBoost** models for Q1 2019, including interactive Plotly charts with 95% confidence bands and model metrics (RMSE, MAE, MAPE, $R^2$).
3. **🗂️ Category Analysis:** Granular product-level forecasts for key departments (Furniture, Office Supplies, Technology) highlighting YoY changes.
4. **🗺️ Region Analysis:** Spatial sales forecasts split across Central, East, South, and West markets, detailing allocation weights and demand growth ranks.
5. **🛡️ Anomaly Center:** Weekly outlier detector using **Isolation Forest** (machine learning) and rolling **Z-score** (statistical) models to flag operational shocks.
6. **🎯 Segmentation:** Product clustering utilizing **K-Means** and **Principal Component Analysis (PCA)** to categorize sub-categories based on volume, growth, and volatility.
7. **💼 Executive Summary & PDF:** Structured business recommendations derived from analytical findings, with an embedded automated **Financial Waterfall chart** and exportable **PDF report generator**.

---

## 🏗️ Repository Structure

*   `app.py`: Self-contained Streamlit application housing the entire UI layout, data loader, interactive components, and lazy-loaded training pipelines.
*   `requirements.txt`: Python package pins optimized for environment stability (pins correct versions of `streamlit`, `prophet`, `cmdstanpy`, `pandas`, `scipy`, etc.).
*   `packages.txt`: System-level dependencies (`gcc`, `build-essential`) needed to compile C++ Stan compiler tools in deployed Linux containers.
*   `runtime.txt`: Python version specification pinning the environment to `python-3.11` (ensuring compatibility with Cython extensions in Prophet).
*   `train.csv`: Underlying dataset containing retail transaction records from 2015–2018.
*   `charts/`: Pre-rendered baseline analytical charts used for quick rendering and backup visualizations.

---

## ⚙️ How We Solved Deployed Crash Risks (Streamlit Cloud Production-Ready)

To run successfully in a restricted cloud environment (e.g., Streamlit Community Cloud with a strict **1 GB RAM limit**), the code underwent key architectural adjustments:

1. **Lazy Loading of Heavy Libraries:** 
   Imports of heavy compiled libraries (`prophet`, `cmdstanpy`, `scikit-learn`) are deferred inside functions and executed only when a user visits a page requesting those features. This keeps the application container's startup footprint light (~250 MB instead of ~750 MB), preventing Out-of-Memory (OOM) crashes.
2. **Deferred CmdStan Compilation:** 
   The download and setup of the C++ CmdStan solver is run lazily inside the cached training function instead of app startup. This prevents Streamlit's initial health check from timing out with an `EOF` error.
3. **Strict Version Alignment:** 
   Pinned `cmdstanpy==1.2.2` and `pyarrow==15.0.2` to resolve low-level serialization segmentation faults and API signature conflicts on Linux containers.
4. **Thread-Safety Constraints:**
   Limited OpenBLAS, MKL, and OpenMP to `1` thread at the very top of `app.py` to prevent CPU over-subscription and native multithreading conflicts.
5. **Robust Exception Catching:**
   All machine learning operations are wrapped in `try/except` blocks to display descriptive warning prompts instead of letting backend failures crash the user interface.

---

## 💻 Local Setup & Execution

### Prerequisites
*   Python 3.11 (Recommended)
*   A C++ compiler (GCC / Clang / MSVC) installed on your system PATH

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ajaikumar18/SalesForecasting_AjaiKumar.git
    cd SalesForecasting_AjaiKumar
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Dashboard:**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Models & Machine Learning Architecture

*   **Facebook Prophet:** Configured with custom changepoint scales (`0.01`), yearly seasonality, and weekly seasonality. Chosen as the primary forecast model due to its robust capture of strong Q4 seasonal trends and weekly corporate shipping cycles (achieving a **16.04% MAPE**).
*   **SARIMA:** Fits a seasonal autoregressive integrated moving average configuration to serve as a baseline statistical comparison.
*   **XGBoost:** Trains an ensemble gradient boosted regressor mapping time lags and rolling features.
*   **Isolation Forest:** Uses unsupervised tree splits (with `4%` contamination rate) to isolate multidimensional sales/deviation outliers.
*   **K-Means & PCA:** Clusters products into 4 categories (Core Volume, Growing Demand, High-Risk Volatile, Low-Velocity) based on volume, AOV, volatility, and YoY growth, projected onto a 2D plane capturing `97%` variance.
