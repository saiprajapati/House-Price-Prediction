# 🏠 House Price Predictor — ML Project

A professional end-to-end machine learning pipeline that predicts residential house prices based on structural and amenity features using multiple regression algorithms.

---

## 📁 Project Structure

```
house-price-predictor/
├── data/
│   └── house_data.csv          # Dataset (545 records, 13 features)
├── models/
│   ├── best_model.pkl          # Saved best model (auto-generated)
│   ├── scaler.pkl              # Fitted StandardScaler (auto-generated)
│   └── model_results.csv       # Comparison metrics (auto-generated)
├── plots/                      # All EDA + evaluation plots (auto-generated)
├── src/
│   ├── preprocess.py           # Data loading, encoding, scaling, splitting
│   ├── train.py                # Multi-model training
│   ├── evaluate.py             # Metrics computation & comparison
│   ├── visualize.py            # All EDA and model evaluation plots
│   └── predict.py              # Inference on new input (CLI-ready)
├── main.py                     # 🚀 Full pipeline entrypoint
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

**Source:** `data/house_data.csv` — 545 rows, 13 columns

| Feature | Type | Description |
|---|---|---|
| `price` | int | Target — House price in ₹ |
| `area` | int | Area in sq ft |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | int | Number of bathrooms |
| `stories` | int | Number of floors |
| `mainroad` | yes/no | Adjacent to main road |
| `guestroom` | yes/no | Has guest room |
| `basement` | yes/no | Has basement |
| `hotwaterheating` | yes/no | Has hot water heating |
| `airconditioning` | yes/no | Has AC |
| `parking` | int | Parking spaces |
| `prefarea` | yes/no | In preferred area |
| `furnishingstatus` | categorical | furnished / semi-furnished / unfurnished |

---

## ⚙️ Setup

### 1. Clone and install

```bash
git clone https://github.com/saiprajapati/house-price-predictor.git
cd house-price-predictor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
- Generate 8 EDA + evaluation plots in `plots/`
- Train 6 regression models
- Print a comparison report
- Save the best model to `models/best_model.pkl`

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularisation (α=10) |
| Lasso Regression | L1 regularisation (α=1000) |
| Decision Tree | max_depth=6 |
| Random Forest | 100 estimators |
| **Gradient Boosting** | **Best — 100 estimators** |

**Best model results:**

| Metric | Value |
|---|---|
| R² | 0.666 |
| MAE | ₹960,579 |
| RMSE | ₹1,299,761 |
| MAPE | 20.77% |

---

## 🔮 Predict on New Data

```bash
python src/predict.py \
  --area 6000 --bedrooms 3 --bathrooms 2 --stories 2 \
  --mainroad yes --guestroom no --basement no \
  --hotwaterheating no --airconditioning yes \
  --parking 1 --prefarea yes --furnishingstatus furnished
```

Output:
```
🏠  Predicted House Price: ₹ 5,877,000
```

Or import in Python:

```python
from src.predict import predict

price = predict(
    model_path="models/best_model.pkl",
    scaler_path="models/scaler.pkl",
    area=6000, bedrooms=3, bathrooms=2, stories=2,
    mainroad="yes", guestroom="no", basement="no",
    hotwaterheating="no", airconditioning="yes",
    parking=1, prefarea="yes", furnishingstatus="furnished"
)
print(f"₹ {price:,.0f}")
```

---

## 📈 Plots Generated

| File | Description |
|---|---|
| `01_price_distribution.png` | Histogram + box plot of target |
| `02_correlation_heatmap.png` | Pearson correlation matrix |
| `03_features_vs_price.png` | Scatter plots — numeric features vs price |
| `04_categorical_impact.png` | Box plots — binary features vs price |
| `05_model_comparison.png` | Side-by-side bar chart of all metrics |
| `06_actual_vs_predicted.png` | Scatter + residual plot for best model |
| `07_feature_importance.png` | Feature importances from best model |
| `08_all_model_predictions.png` | Actual vs predicted grid for all models |

---

## 🛠 Tech Stack

- **Python 3.10+**
- **pandas** — Data manipulation
- **scikit-learn** — ML models, preprocessing, metrics
- **matplotlib / seaborn** — Visualisation
- **joblib** — Model persistence

---

## 👤 Author

**Sai Prajapati**
- GitHub: [@saiprajapati](https://github.com/saiprajapati)
- LinkedIn: [sai-prajapati](https://linkedin.com/in/sai-prajapati)
- Portfolio: [sai-portfolio-chi-six.vercel.app](https://sai-portfolio-chi-six.vercel.app)
