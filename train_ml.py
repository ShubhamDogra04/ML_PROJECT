import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare(train_csv, val_csv=None):
    df_train = pd.read_csv(train_csv)
    X_train = df_train.iloc[:, 2:].values
    y_train = LabelEncoder().fit_transform(df_train['label'].values)

    if val_csv:
        df_val = pd.read_csv(val_csv)
        X_val = df_val.iloc[:, 2:].values
        y_val = LabelEncoder().fit_transform(df_val['label'].values)
    else:
        X_val, y_val = None, None

    return X_train, y_train, X_val, y_val

def fit_and_eval(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = None

    svm = SVC(kernel='rbf')
    svm.fit(X_train_scaled, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    models = [("SVM", svm), ("KNN", knn)]

    for name, model in models:
        if X_val_scaled is None:
            continue

        preds = model.predict(X_val_scaled)
        print(f"\n=== {name} Results ===\n")
        print(classification_report(y_val, preds))

        cm = confusion_matrix(y_val, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=False)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val = load_and_prepare(args.train_csv, args.val_csv)
    fit_and_eval(X_train, y_train, X_val, y_val)
