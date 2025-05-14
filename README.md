This project uses the classic Iris flower dataset to build and compare two machine learning models:
A Random Forest Classifier
A Bagging Classifier with Linear Support Vector Classifier (LinearSVC) as the base estimator

Both models are optimized using grid search with cross-validation and evaluated using metrics such as OOB (Out-of-Bag) score, training accuracy, and confusion matrices.

The Iris dataset is a well-known multi-class classification dataset consisting of 150 flower samples across three species:

Iris-setosa, Iris-versicolor, Iris-virginica

Each flower is described by four numeric features:

SepalLengthCm	Sepal length in cm
SepalWidthCm	Sepal width in cm
PetalLengthCm	Petal length in cm
PetalWidthCm	Petal width in cm

There are 50 samples per class, and the balanced dataset makes it ideal for comparing classification algorithms.


Random Forest
Training Accuracy: Very high (close to 100%)
OOB Score: Also high, suggesting the model generalizes well
Feature Importances: Petal length and width are the most important features
Confusion Matrix: Shows excellent class separation; few to no misclassifications

Bagging SVC
Training Accuracy: High, though slightly below Random Forest
OOB Score: Also strong, validating generalization
Confusion Matrix: Shows occasional confusion between versicolor and virginica, but still strong overall

