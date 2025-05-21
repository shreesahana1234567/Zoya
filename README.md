# 📦 Supply Chain Management System – Machine Learning Based

## 🧾 Objective

Optimize and forecast various components of supply chain management—such as demand forecasting, inventory optimization, and supplier performance evaluation—using advanced machine learning techniques.

---

## 🚀 Key Features & Functionalities

* 🔍 **Demand Forecasting**
  Predict future demand using time-series models (Prophet) and regression algorithms (Linear Regression, Random Forest).

* 📦 **Inventory Optimization**
  Maintain optimal inventory levels to minimize holding and shortage costs.

* 🛒 **Supplier Performance Evaluation**
  Rank and evaluate suppliers based on delivery times, product quality, and compliance metrics.

* 📊 **Interactive Dashboard**
  Visualize KPIs and analytics through real-time, user-friendly dashboards built with Streamlit.

* ⚠ **Supply Chain Risk Prediction**
  Classify and detect potential supply chain disruptions using models like XGBoost.

---

## 🧰 Technology Stack

| Component           | Technology                                               |
| ------------------- | -------------------------------------------------------- |
| **Languages**       | Python                                                   |
| **Libraries**       | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly |
| **ML Frameworks**   | Scikit-learn, XGBoost, Prophet                           |
| **Visualization**   | Streamlit                                                |
| **Data Storage**    | CSV, SQLite (optional)                                   |
| **Version Control** | Git, GitHub                                              |

---

## ⚙ Workflow

1. **Data Collection**
   Import datasets (e.g., sales, inventory, supplier logs) from Kaggle or synthetic generation.

2. **Preprocessing**
   Clean, normalize, and engineer features to prepare for model training.

3. **Model Training**
   Apply ML models for forecasting, classification, and clustering tasks.

4. **Evaluation**
   Use metrics like RMSE, MAE, accuracy, and F1-score to assess performance.

5. **Visualization**
   Present insights and predictions via an interactive Streamlit dashboard.

6. **Prediction**
   Generate live predictions through user inputs on the dashboard.

---

## 📥 Data Collection

* **Source**: Kaggle Supply Chain Dataset, synthetic data generation.
* **Fields**: Order date, product type, inventory level, delivery time, supplier score.
* **Access**: Downloaded via Kaggle API and processed using Python scripts.

---

## 🎮 Interactive Controls

* Select product ID, region, and date range for forecasting.
* Filter suppliers by score threshold.
* Adjust inventory reorder thresholds.
* Real-time form inputs for risk and demand predictions.

---

## 🧠 Machine Learning Techniques

* **Linear Regression, Random Forest** – Demand forecasting.
* **XGBoost Classifier** – Risk prediction for disruptions.
* **K-Means Clustering** – Supplier segmentation and grouping.
* **Prophet** – Time-series forecasting for trends and seasonality.
* **PCA (Optional)** – Dimensionality reduction for high-dimensional data.

---

## 🏋 Model Training Details

* **Train/Test Split**: 80/20
* **Cross-Validation**: 5-fold for hyperparameter tuning
* **Feature Importance**: Visualized using SHAP values and model-specific importance plots

---

## 📤 Output Overview

* 📈 **Forecast Graphs**: Visual comparison of predicted vs. actual values
* ⚠ **Risk Alerts**: "At Risk" / "Stable" labels for disruptions
* 🏅 **Supplier Ranking**: Normalized scores (0–100) based on performance
* 💹 **Inventory Health Score**: Composite metric based on holding cost, lead time, and shortage risk

---

## 🔮 Future Enhancements

* API integration for **real-time data streaming**
* Incorporation of **anomaly detection** for logistics monitoring
* **Production deployment** using Docker and FastAPI

---


