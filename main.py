from data_preprocessing import load_and_prepare_data, create_lagged_features, prepare_lead_data
from model_training import train_model, find_best_alpha, evaluate_model
import pandas as pd
import numpy as np

def main():
    train_file = 'train_data_pca.mat'
    test_file = 'test_data_pca.mat'
    columns_to_use = [4, 5, 11, 12, 17]
    
    df_train, mat_test = load_and_prepare_data(train_file, test_file, columns_to_use)
    df_train_with_lags = create_lagged_features(df_train)
    df_train_lead_1 = prepare_lead_data(df_train_with_lags, 1)

    X_train = df_train_lead_1.iloc[:, :-1].values
    y_train = df_train_lead_1.iloc[:, -1].values

    # Find best alpha
    alphas = np.linspace(0.001, 2, 200)
    best_params = find_best_alpha(X_train, y_train, alphas)
    print(f"Best Alpha: {best_params['Alpha']}, R²: {best_params['R² Score']}")

    # Train with best alpha
    model = train_model(X_train, y_train, best_params['Alpha'])

    # Prepare test data
    X_test = np.zeros((24, 30))  # Assuming 30 features
    Y_test = np.zeros(24)
    for i in range(24):
        test_data = mat_test[i][:, columns_to_use]
        test_data[:, -1] = (test_data[:, -1] - np.mean(test_data[:, -1])) / np.std(test_data[:, -1])
        df_test = pd.DataFrame(test_data, columns=['oniw', 'pmmw', 'oni', 'pmm', 'GPP'])
        df_test_with_lags = create_lagged_features(df_test)
        df_test_lead_1 = prepare_lead_data(df_test_with_lags, 1)
        X_test[i] = df_test_lead_1.iloc[-1, :-1].values
        Y_test[i] = df_test_lead_1.iloc[-1, -1]

    # Evaluate on test set
    test_results = evaluate_model(model, X_test, Y_test, "Lasso Regression")
    print(test_results)

    # Feature Importance
    importance_df = pd.DataFrame(model.coef_, index=df_train_lead_1.columns[:-1], columns=["Coefficient"])
    sorted_importance = importance_df.abs().sort_values('Coefficient', ascending=False)
    print(sorted_importance)

if __name__ == "__main__":
    main()