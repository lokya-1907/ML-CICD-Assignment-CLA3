from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model():
    # Load the Digits dataset
    digits = load_digits()

    X = digits.data
    y = digits.target

    # Split the dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create the Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, X_test, y_test, predictions


if __name__ == "__main__":
    model, accuracy, X_test, y_test, predictions = train_model()

    print("====================================")
    print("Machine Learning Model")
    print("====================================")
    print("Model: Random Forest Classifier")
    print("Dataset: Scikit-learn Digits Dataset")
    print("Task: Handwritten Digit Classification")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Accuracy Percentage: {accuracy * 100:.2f}%")
    print("====================================")
