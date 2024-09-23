import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import randint

credit_card_data = pd.read_csv('/Users/dhruvilshah/Downloads/creditcard_2023.csv')
X = credit_card_data.drop(columns=['Class'])
y = credit_card_data['Class']

# Rebalance the dataset if needed (you can use techniques like oversampling, undersampling, or SMOTE)

# Split the dataset into training and testing sets (considering data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV with stratified cross-validation
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=100,
                                   scoring='accuracy', cv=5, n_jobs=-1, random_state=42)

# Perform randomized search to find the best hyperparameters
random_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = random_search.best_params_
best_rf_model = random_search.best_estimator_

# Predict on the testing set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model on the testing set
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)