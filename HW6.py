import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC


# Load the Iris dataset (train on the entire dataset)

df = pd.read_csv("iris.csv")

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


X = df[features]
y = df["Species"]



# Define the parameter grid for max_depth (range from 2 to 15)
parameters_rf = {
    "max_depth": range(2, 16)
}

# Set up the Random Forest classifier and grid search (using 5-fold CV on the entire dataset)
clf_rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(clf_rf, param_grid=parameters_rf, cv=5)
grid_search_rf.fit(X, y)

# Convert grid search results to a DataFrame and print select columns
results_rf = pd.DataFrame(grid_search_rf.cv_results_)
print(results_rf[['param_max_depth', 'mean_test_score', 'rank_test_score']])
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

# Initialize a RandomForestClassifier with the best max_depth,
# enable OOB scoring, verbose mode, and parallel processing.
rfc = RandomForestClassifier(max_depth=grid_search_rf.best_params_['max_depth'],
                             oob_score=True, verbose=0, n_jobs=-1)
rfc.fit(X, y)

# Print the in-sample (training) score and the OOB score
print(f"Random Forest Train Score: {rfc.score(X, y):.3f}")
print(f"Random Forest OOB Score: {rfc.oob_score_:.3f}")

# Generate and plot the normalized confusion matrix using predictions on the full dataset
cm_rf = confusion_matrix(y, rfc.predict(X), normalize="true")
labels = sorted(y.unique())
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=labels)
disp_rf.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()

importances = pd.DataFrame(rfc.feature_importances_,
                           index=df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].columns,
                           columns=["Importance"])
importances.plot.bar(legend=False)
plt.ylabel("Relative Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------#
param_grid_bag = {
    'estimator__C': [0.1, 1.0, 10.0],
    'max_samples': [0.5, 1.0],
    'n_estimators': [50]  # You can also try [10, 50, 100] here if desired.
}

# Set up the BaggingClassifier with a LinearSVC as the base estimator.
bag = BaggingClassifier(
    estimator=LinearSVC(dual=False, max_iter=10000, random_state=42),
    oob_score=True,
    n_jobs=-1,
    verbose=0,
)

# Grid search with 5-fold cross-validation, refit is True by default.
bag_grid = GridSearchCV(bag, param_grid=param_grid_bag, cv=5)
bag_grid.fit(X, y)

# Convert grid search results to a DataFrame and print selected columns
results_bag = pd.DataFrame(bag_grid.cv_results_)
print(results_bag[[
    "param_estimator__C",
    "param_n_estimators",
    "param_max_samples",
    "mean_test_score",
    "rank_test_score"
]])
print("Best Bagging SVC Parameters:", bag_grid.best_params_)

# Retrieve the best estimator (already refit on entire data)
best_bag = bag_grid.best_estimator_

# Print in-sample (training) score and the OOB score
print(f"Bagging SVC Train Score: {best_bag.score(X, y):.3f}")
print(f"Bagging SVC OOB Score:   {best_bag.oob_score_:.3f}")

# Generate and plot the normalized confusion matrix using predictions on the full dataset
cm_bag = confusion_matrix(y, best_bag.predict(X), normalize="true")
# Get sorted unique labels from the target column
labels = sorted(y.unique())
disp_bag = ConfusionMatrixDisplay(confusion_matrix=cm_bag, display_labels=labels)
disp_bag.plot()
plt.title("Bagging SVC Confusion Matrix")
plt.show()