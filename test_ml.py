import pytest
# add necessary import
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from ml.model import train_model, compute_model_metrics, inference

# implement the first test. Change the function name and input as needed
def test_model_aint_out_here_using_linear_regression():
    """
    Tests that train_model returns a GradientBoostingClassifier instance,
    confirming the expected algorithm is used.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, GradientBoostingClassifier)

# implement the second test. Change the function name and input as needed
def test_inference_spits_out_actual_predictions_not_vibes():
    """
    Tests that the inference function returns a numpy array of predictions
    with the same length as the input data.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_train)


# implement the third test. Change the function name and input as needed
def test_perfect_predictions_get_perfect_scores_shocker():
    """
    Tests that compute_model_metrics returns precision, recall, and F1 of 1.0
    when predictions exactly match the known labels.
    """
    y = np.array([1, 0, 1, 0, 1])
    preds = np.array([1, 0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0