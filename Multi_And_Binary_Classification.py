#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:55:30 2025

@author: sagarshrestha
"""
#importing the data of daibaitic and non daibiatic patient

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Load CSV file into a DataFrame
# data is downloaded from https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/discussion?sort=hotness
df = pd.read_csv("diabetes_prediction_dataset.csv");
#binary classificatoin using svm, K-NN and decision tree. 

#seperating out the feature and lable from the given dataset 
X = df.drop("diabetes", axis = 1)
Y = df["diabetes"] #target

# seperating out the categorical and numerical datatype
categorical_cols = ["gender","smoking_history"];
numeric_cols = [col for col in X.columns if col not in categorical_cols]

#preprocesssing for numerical and categorical data

preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), numeric_cols),               # scale numeric features
        ("cat", OneHotEncoder(drop="first"), categorical_cols) # encode categorical features
])

#splitiing the data such that 75% of data is used for training the model and 25% is used as test data 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.25, random_state=2)

# fit and transform the train data, transform the test data 
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Dictionary to store results
results = []

# SVM models
svm_kernels = ["linear", "rbf"]
for kernel in svm_kernels:
    model = SVC(kernel=kernel, C=1.0, gamma="scale")
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": f"SVM ({kernel})", "Accuracy": acc})

# KNN models
knn_neighbors = [2, 3, 4, 6, 8, 10]
for k in knn_neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": f"KNN (k={k})", "Accuracy": acc})

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_processed, y_train)
y_pred = dt_model.predict(X_test_processed)
acc = accuracy_score(y_test, y_pred)
results.append({"Model": "Decision Tree", "Accuracy": acc})

# Create results table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("\n=== Model Accuracy Summary ===")
print(results_df)

# # Building the pipeline with SVM with linear kernel
# pipeline_SVM_linear = Pipeline([("preprocessor",preprocessor), ("classifier", SVC(kernel="linear"))]);

# #Building the pipeline with svm with rbf as kernel
# pipeline_SVM_RBF = Pipeline([("preprocessor", preprocessor), ("classifier", SVC(kernel = "rbf"))]);

# #train the model with kernel as linear for SVM algorithm
# pipeline_SVM_linear.fit(X_train, Y_train);

# #train the model with kernel as rbf
# pipeline_SVM_RBF.fit(X_train, Y_train);
# # Make predictions on test set on svm with linear kernel
# Y_pred_SVM_linear = pipeline_SVM_linear.predict(X_test)

# # make the prediction on test set on svm with rbf kernel
# Y_pred_SVM_rbf = pipeline_SVM_RBF.predict(X_test);

# # Evaluate the model
# accuracy_with_linear_kernel = accuracy_score(Y_test, Y_pred_SVM_linear);
# accuracy_with_rbf_kernel = accuracy_score(Y_test, Y_pred_SVM_rbf);

# print("Accuracy in SVM with linear kernel:", accuracy_with_linear_kernel)

# print("Accuracy in SVM with linear kernel:", accuracy_with_rbf_kernel)

# # # Predictions
# # y_pred = model.predict(X_test)
# # print("Accuracy:", accuracy_score(y_test, y_pred))