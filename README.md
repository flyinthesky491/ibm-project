# 🏠 House Price Prediction

A full-stack machine-learning project that predicts residential house prices using a **Ridge Regression** model, served by a **Flask REST API** and explored through a **Streamlit** frontend.

---

## Project Structure

```
house_price_project/
├── backend/
│   └── app.py                  # Flask REST API
├── data/
│   ├── generate_data.py        # Script to create house_price.csv
│   └── house_price.csv         # Dataset (generated)
├── frontend/
│   └── ui.py                   # Streamlit UI (4 pages)
├── model/
│   └── house_price_model.pkl   # Trained model (generated)
├── report_images/              # Training plots (generated)
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   ├── feature_importance.png
│   └── price_distribution.png
├── generate_report.py          # HTML report generator
├── train_model.py              # Model training script
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python data/generate_data.py

# 3. Train the model (saves model + report images)
python train_model.py

# 4. Start Flask API  (Terminal 1)
python backend/app.py

# 5. Start Streamlit UI  (Terminal 2)
streamlit run frontend/ui.py

# 6. (Optional) Generate standalone HTML report
python generate_report.py
```

---

## Frontend Pages

| Page | Description |
|------|-------------|
| 🔮 Predict Price | Interactive form → calls `/api/predict` |
| 📊 Dataset Explorer | Price histogram, box plots, correlation heatmap, raw data table |
| 📈 Model Performance | R²/MAE/RMSE metrics + 4 training plots |
| ℹ️ About | Project overview and run instructions |

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/health` | Service health check |
| GET  | `/api/model-info` | Model metrics & feature list |
| POST | `/api/predict` | Predict price from JSON body |
| GET  | `/api/dataset-stats` | Summary statistics of the dataset |

### Example POST `/api/predict`
```json
{
  "bedrooms": 3, "bathrooms": 2, "sqft_living": 1800,
  "sqft_lot": 5000, "floors": 1, "waterfront": 0,
  "view": 0, "condition": 3, "grade": 7,
  "yr_built": 1990, "yr_renovated": 0,
  "lat": 47.5, "long": -122.0
}
```

---

## Model Performance (typical)

| Metric | Value      |
|--------|------------|
| R²     | ~0.73      |
| MAE    | ~$40,000   |
| RMSE   | ~$55,000   |

---

## Tech Stack

| Layer    | Technology                      |
|----------|---------------------------------|
| ML       | scikit-learn, numpy             |
| Backend  | Flask, Flask-CORS               |
| Frontend | Streamlit, Plotly               |
| Data     | pandas, matplotlib, seaborn     |
