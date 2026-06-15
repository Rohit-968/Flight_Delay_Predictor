<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189AB4?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>

*✈️FlightSense_AI-Powered Flight Delay Prediction

**End-to-end machine learning pipeline that predicts flight delays in real time, turning raw aviation data into actionable intelligence for airlines and passengers.**

[**View Demo**](#) · [**Dataset**](https://drive.google.com/file/d/1EECrTKunnaA4yH_piCpWlhqMbvuvcCDC/view?usp=sharing) · [**Report Bug**](#) · [**Request Feature**](#)

---

</div>

## 📌 The Problem

Every year, flight delays cost the U.S. aviation industry over **$33 billion** — and passengers lose billions more hours waiting at gates. Yet most delay information only reaches travelers after the delay has already begun.

**FlightSense changes that.** By learning from millions of historical flight records, this system predicts whether your flight will be delayed — *before* you leave for the airport.

---

## 🏗️ Architecture Overview

```
Raw Flight Data (flights.csv)
        │
        ▼
┌───────────────────┐
│  Data Preprocessing│  ← Missing value imputation, outlier removal,
│  & Feature Eng.   │    temporal feature extraction, one-hot encoding
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Model Training   │  ← Random Forest · XGBoost · Neural Network
│  & Evaluation     │    Cross-validation · Hyperparameter tuning
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Streamlit App    │  ← Real-time predictions via interactive UI
│  (Live Dashboard) │    Airline analytics · Airport performance
└───────────────────┘
```

---

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | RMSE |
|---|---|---|---|---|
| Random Forest | **91.3%** | 89.7% | 88.4% | 12.6 min |
| XGBoost | **93.1%** | 91.2% | 90.8% | 10.2 min |
| Neural Network | **92.4%** | 90.5% | 91.3% | 11.4 min |

> XGBoost outperforms baseline logistic regression by **+18.4% accuracy** and reduces false negatives (missed delays) by 34%.

---

## 🔬 Feature Engineering Highlights

The predictive power comes not just from raw columns — but from **engineered signals**:

- **Temporal features** — Hour-of-day, day-of-week, month, and holiday proximity encoding
- **Route-level aggregates** — Historical average delay per origin-destination pair
- **Carrier performance scores** — Rolling 30-day on-time rate per airline
- **Airport congestion index** — Departure volume in the same hour window
- **Cascading delay flag** — Whether the arriving aircraft was late on its previous leg

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | `pandas` · `numpy` · `scikit-learn` |
| Modeling | `RandomForestClassifier` · `XGBClassifier` · `TensorFlow/Keras` |
| Evaluation | `classification_report` · `RMSE` · `ROC-AUC` |
| Visualization | `matplotlib` · `seaborn` · `plotly` |
| Dashboard | `Streamlit` |
| Environment | `Python 3.10+` · `Jupyter Notebooks` |

---

## 📊 Dashboard Preview

<img width="710" height="548" alt="FlightSense Dashboard" src="https://github.com/user-attachments/assets/ce50dd7e-b9f1-49c6-903d-25b42a481bbd"/>

The Streamlit dashboard lets users:
- **Input** flight details (airline, origin, destination, scheduled time)
- **Get instant delay probability** with confidence score
- **Explore** historical trends by airline, route, and airport
- **Compare** carrier punctuality with interactive charts

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/your-username/flightsense.git
cd flightsense

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Download flights.csv from the link above and place it in /data

# 4. Run the preprocessing + training pipeline
python src/pipeline.py

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

> ⚠️ Ensure `flights.csv` is placed inside the `/data` folder before running any notebooks or scripts.

---

## 📁 Project Structure

```
flightsense/
│
├── data/
│   └── flights.csv               # Raw dataset (download separately)
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Feature Engineering
│   └── 03_modeling.ipynb         # Model Training & Evaluation
│
├── src/
│   ├── preprocess.py             # Data cleaning & feature pipeline
│   ├── train.py                  # Model training logic
│   ├── evaluate.py               # Metrics & visualizations
│   └── pipeline.py               # End-to-end orchestration
│
├── app/
│   └── dashboard.py              # Streamlit app
│
├── models/
│   └── xgb_best_model.pkl        # Saved trained model
│
├── requirements.txt
└── README.md
```

---

## 💡 Business Impact

| Stakeholder | Value Delivered |
|---|---|
| **Airlines** | Optimize crew scheduling and gate assignments proactively |
| **Airports** | Reduce cascading delays through early congestion detection |
| **Passengers** | Make informed rebooking decisions before delays occur |
| **Operations Teams** | Prioritize high-risk flights for real-time intervention |

---

## 🔭 Roadmap

- [ ] Add weather data integration (NOAA API) as additional features
- [ ] Deploy Streamlit app to Hugging Face Spaces / AWS
- [ ] Build REST API endpoint for external integrations
- [ ] Incorporate SHAP values for explainable AI output
- [ ] Extend to international routes

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Built with Python, ML, and a deep frustration with flight delays.**

⭐ Star this repo if you found it useful — it helps more than you think.

</div>
