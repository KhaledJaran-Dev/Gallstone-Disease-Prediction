# Gallstone Disease Prediction

End-to-end ML classification pipeline to predict gallstone disease using the UCI dataset.

## Methods
- Sequential Feature Selection (top 10 features)
- 5 classifiers: Logistic Regression, SVM, Decision Tree, Random Forest, Gradient Boosting
- Hyperparameter tuning with GridSearchCV + 5-fold cross-validation
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Overfitting analysis via train-test accuracy gap
- Model explainability: SHAP + LIME

## Tools
Python, scikit-learn, pandas, NumPy, SHAP, LIME
