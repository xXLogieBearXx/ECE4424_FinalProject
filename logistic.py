import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('messidor_features.csv')  # Assuming the file is in the same folder

# Split the data into features (X) and target variable (y)
X = data.iloc[:, :-1]  # Features are all columns except the last one
y = data.iloc[:, -1]   # Target variable is the last column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#class_weights = {0: 1, 1: 2.5}  # uncomment for more weight for class 1
class_weights = {0: 1, 1: 1}  # uncomment fot equal weight for class 1

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Initialize the logistic regression model
log_reg_model = LogisticRegression(class_weight=class_weights, max_iter=5000)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=log_reg_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Train the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions and calculate probabilities
y_pred = best_model.predict(X_test)
y_probabilities = best_model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Track the count of incorrect predictions where predicted class is 0
incorrect_count_0 = 0
incorrect_count_1 = 0

# Report confidence (probability) of every wrong prediction
for i, (pred, true_label, probs) in enumerate(zip(y_pred, y_test, y_probabilities)):
    if pred != true_label:
        #print(f"Sample {i}: Predicted class: {pred}, True class: {true_label}")
        #print(f"Probabilities for class 0: {probs[0]}, Probabilities for class 1: {probs[1]}")
        #print()

        # Increment count if predicted class is 0 and it's wrong
        if pred == 0:
            incorrect_count_0 += 1
        if pred == 1:
            incorrect_count_1 += 1

# Print the count of incorrect predictions where predicted class is 0
print(f"Number of times predicted class 0 was wrong: {incorrect_count_0}")
print(f"Number of times predicted class 1 was wrong: {incorrect_count_1}")