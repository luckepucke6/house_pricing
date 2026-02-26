## Production-Style ML Price Prediction API

---

# 🎯 PROJECT GOAL

Build a structured ML project using mostly notebooks, but in a way that mimics real MLOps workflows.

You will:
- Explore data in a dedicated EDA notebook
- Train and evaluate models in structured notebooks
- Save artifacts (model + metrics)
- Expose the model through a minimal API
- Keep a clean, professional project structure

This simulates how real ML teams work.

---

# 🧠 BIG PICTURE

01_eda.ipynb  
↓  
02_training.ipynb  
↓  
03_evaluation.ipynb  
↓  
Save model.pkl  
↓  
FastAPI loads model  

Research (Notebooks) → Inference (API layer)

---

# 📅 WEEK 1 – EDA

## Notebook: notebooks/01_eda.ipynb

### Objectives
- Load dataset
- Understand features
- Identify target variable
- Visualize relationships
- Detect missing values and outliers

### Implement
- df.head()
- df.info()
- df.describe()
- Correlation heatmap
- Scatter plots
- Distribution plots

### Search For
- pandas dataframe exploration
- seaborn heatmap correlation
- detect outliers python
- feature vs target explanation
- data leakage explained

### Concepts To Understand
- Feature
- Target
- Correlation
- Distribution
- Data leakage

---

# 📅 WEEK 2 – TRAINING

## Notebook: notebooks/02_training.ipynb

### Structure

1. Load data
2. Select features (X)
3. Select target (y)
4. Train/test split
5. Train model
6. Evaluate model
7. Save model

### Example Code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

X = df[["size", "rooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)

joblib.dump(model, "../models/model.pkl")
````

### Search For

* train test split sklearn
* linear regression sklearn example
* random forest regression example
* mean absolute error explained
* joblib save model python

### Concepts To Understand

* Overfitting
* Underfitting
* MAE
* RMSE
* Reproducibility (random_state)

---

# 📅 WEEK 3 – MODEL COMPARISON & EXPERIMENTS

## Notebook: notebooks/03_evaluation.ipynb

### Objectives

* Train multiple models
* Compare metrics
* Save experiment results

### Try

* Linear Regression
* Random Forest
* Gradient Boosting

### Save Results

```python
results.to_csv("../metrics/experiments.csv", index=False)
```

### Search For

* cross validation sklearn
* bias vs variance explained
* model comparison sklearn
* regression metrics comparison

### Concepts To Understand

* Cross-validation
* Model selection
* Bias vs variance
* Why simple models sometimes win

---

# 📅 WEEK 4 – API LAYER (Production Bridge)

## File: app/main.py

### Objective

* Load model.pkl
* Expose prediction endpoint

### Example

```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(size: float, rooms: int):
    prediction = model.predict(np.array([[size, rooms]]))
    return {"predicted_price": float(prediction[0])}
```

### Run API

uvicorn app.main:app --reload

### Search For

* FastAPI beginner tutorial
* uvicorn run fastapi
* REST API python example
* dockerize fastapi app

### Concepts To Understand

* Training vs inference
* Model serialization
* REST API
* Why inference should be lightweight

---

# 📁 FINAL PROJECT STRUCTURE

ml-project/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── models/
│   └── model.pkl
│
├── metrics/
│   └── experiments.csv
│
├── app/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

# 🔥 WHAT MAKES THIS MLOps-LIKE?

✔ Separation between research and inference
✔ Saved artifacts (model + metrics)
✔ Reproducibility
✔ Clean structure
✔ Deployable API layer

---

# 🧱 OPTIONAL EXTENSIONS (AFTER WEEK 4)

* Add logging
* Add unit tests with pytest
* Add Docker
* Add DVC for versioning
* Add CI/CD with GitHub Actions
* Deploy to Render or Fly.io

---

# 🧠 REFLECTION QUESTIONS

1. Why should training NOT happen inside the API?
2. Why do we save model.pkl instead of retraining each time?
3. What makes this project production-oriented?
4. What part is research and what part is inference?

---

# 🎯 END RESULT

You will have:

* A structured ML workflow
* Saved experiments
* A deployable API
* A clean GitHub-ready project
* A strong talking point for LIA and interviews