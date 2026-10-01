from model import train_model


def test_model_training():
    model, accuracy, X_test, y_test, predictions = train_model()

    # Check that the model was created
    assert model is not None

    # Check that accuracy is valid
    assert 0.0 <= accuracy <= 1.0

    # Check that predictions were generated
    assert len(predictions) == len(y_test)


def test_model_accuracy():
    model, accuracy, X_test, y_test, predictions = train_model()

    # The Random Forest model should achieve
    # at least 90% accuracy on the Digits dataset
    assert accuracy >= 0.90


def test_prediction_output():
    model, accuracy, X_test, y_test, predictions = train_model()

    # Check that predictions contain valid digits
    assert all(0 <= prediction <= 9 for prediction in predictions)
