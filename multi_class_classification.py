import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier

# https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data
df = pd.read_csv("mobile_price_dataset.csv")

#segregrating the fetures and label 
X = df.drop("price_range", axis = 1)
Y= df["price_range"]

# Splitting the data such that test data is 25 % of the whole dataset 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1);

# preprocessing training dataset 
scaler = StandardScaler();
X_train= scaler.fit_transform(X_train);
X_test = scaler.transform(X_test);

# 4. Train and evaluate different kernels
kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    print(f"\n=== SVM with {kernel} kernel ===")
    svm_clf = SVC(kernel=kernel, decision_function_shape='ovr', random_state=42)
    svm_clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = svm_clf.predict(X_test)
    
    # Evaluation
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


# List of n_neighbors to test
neighbors_list = [2, 3, 4, 6, 8, 10]

for k in neighbors_list:
    print(f"\n=== KNN with n_neighbors = {k} ===")
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = knn_clf.predict(X_test)
    
    # Evaluation
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    
    # Train Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# Predictions
y_pred = dt_clf.predict(X_test)

# Evaluation
print("=== Decision Tree ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))