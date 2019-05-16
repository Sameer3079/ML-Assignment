import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load the data set
df = pd.read_csv("./data/adult.csv")

# Replace unknown values with Nan
df = df.replace('?', np.NaN)

# Drop samples which contain Nan data
df = df.dropna()

# Map the results of the samples to numerical values
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Compressing unnecessary data variations
df['marital.status'] = df['marital.status'].replace(['Widowed', 'Divorced', 'Separated', 'Never-married'], 'single')

df['marital.status'] = df['marital.status'].replace(
    ['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse'], 'married')

# Get columns which contain object data
categorical_df = df.select_dtypes(include=['object'])

# Load the label encoder
enc = LabelEncoder()

categorical_df = categorical_df.apply(enc.fit_transform)

# Removes the object data columns from the data set
df = df.drop(categorical_df.columns, axis=1)
# Adds the removed columns with the encoded data
df = pd.concat([df, categorical_df], axis=1)

# Separate inputs and outputs in the data set
X = df.drop('income', axis=1)
y = df['income']

# Split the data set into training data set and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

##---
clf = DecisionTreeClassifier(random_state=24)
clf.fit(X_train, y_train)

# Get predictions for the test input data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree accuracy = ", accuracy * 100)
# print(classification_report(y_test, y_pred))

#---
clf = RandomForestClassifier(n_estimators=100, random_state=24)
clf.fit(X_train, y_train)

# Get predictions for the test input data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(" n_estimators = 100, Random Forests accuracy = ", accuracy * 100)
# print(classification_report(y_test, y_pred))
##---
clf = RandomForestClassifier(n_estimators=1000, random_state=24)
clf.fit(X_train, y_train)

# Get predictions for the test input data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(" n_estimators = 1000, Random Forests accuracy = ", accuracy * 100)
# print(classification_report(y_test, y_pred))
#-----
clf = AdaBoostClassifier(n_estimators=100, random_state=24)
clf.fit(X_train, y_train)

# Get predictions for the test input data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("n_estimators=100, Ada Boost accuracy", accuracy * 100)
# print(classification_report(y_test, y_pred))

#-----
clf = AdaBoostClassifier(n_estimators=1000, random_state=24)
clf.fit(X_train, y_train)

# Get predictions for the test input data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("n_estimators=1000, Ada Boost accuracy", accuracy * 100)
# print(classification_report(y_test, y_pred))

#-----
clf = AdaBoostClassifier(n_estimators=10000, random_state=24)
clf.fit(X_train, y_train)

# Get predictions for the test input data
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("n_estimators=10000, Ada Boost accuracy", accuracy * 100)
# print(classification_report(y_test, y_pred))