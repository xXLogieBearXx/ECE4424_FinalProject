import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Load the data
data = pd.read_csv('messidor_features.csv')
# Split the data into features (X) and target variable (y)
X = data.iloc[:, :-1]  # Features are all columns except the last one
y = data.iloc[:, -1]   # Target variable is the last column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
# class_weights = {0: 1, 1: 2.5}  # uncomment for more weight for class 1
class_weights = {0: 1, 1: 1}  # uncomment for equal weight for class 1

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

# Get results as DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Plotting accuracies for various combinations of hyperparameters
plt.figure(figsize=(10, 6))

for solver in ['liblinear', 'saga']:
    for penalty in ['l1', 'l2']:
        params = results[(results['param_solver'] == solver) & (results['param_penalty'] == penalty)]
        plt.plot(params['param_C'], params['mean_test_score'], marker='o', label=f'{solver} - {penalty}')

plt.title('Grid Search Results')
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Mean Test Accuracy')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions and calculate probabilities for training and test sets
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Calculate training and test accuracies
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Calculate the loss function (log loss)
loss_train = log_loss(y_train, best_model.predict_proba(X_train))
loss_test = log_loss(y_test, best_model.predict_proba(X_test))

print("Training Accuracy:", accuracy_train)
print("Log Loss Train:", loss_train)
print("Test Accuracy:", accuracy_test)
print("Log Loss Test:", loss_test)

# Track the count of incorrect predictions where predicted class is 0
incorrect_count_0 = 0
incorrect_count_1 = 0

# Report confidence (probability) of every wrong prediction in the test set
for i, (pred, true_label, probs) in enumerate(zip(y_pred_test, y_test, best_model.predict_proba(X_test))):
    if pred != true_label:
        # Increment count if predicted class is 0 and it's wrong
        if pred == 0:
            incorrect_count_0 += 1
        if pred == 1:
            incorrect_count_1 += 1

# Print the count of incorrect predictions where predicted class is 0
print(f"Number of times predicted class 0 was wrong: {incorrect_count_0}")
print(f"Number of times predicted class 1 was wrong: {incorrect_count_1}")
