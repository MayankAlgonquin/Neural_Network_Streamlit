# Import accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix

# # Function to predict and evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_pred, y_test)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)

    return accuracy, confusion_mat