import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess_data
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = load_and_preprocess_data('../data/student-mat.csv')

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

with open('../results/model_metrics.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))