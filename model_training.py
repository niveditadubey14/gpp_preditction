from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_model(X_train, y_train, alpha):
    model = Lasso(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return {"Model": name, "RMSE": rmse, "R² Score": r2}

def find_best_alpha(X, y, alphas):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
    results = []
    for alpha in alphas:
        model = train_model(X_train, y_train, alpha)
        val_results = evaluate_model(model, X_val, y_val, "Lasso Regression")
        results.append({**val_results, "Alpha": alpha})
    return max(results, key=lambda x: x["R² Score"])