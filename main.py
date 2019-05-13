import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data set
df = pd.read_csv("./data/adult.csv")

# Replace unknown values with null
df = df.replace('?', np.NaN)

# Drop samples which contain Nan data
df = df.dropna()

# Map the results of the samples to numerical values
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# Compressing unnecessary data variations
df['marital.status'] = df['marital.status'].replace(['Widowed', 'Divorced', 'Separated', 'Never-married'], 'single')

df['marital.status'] = df['marital.status'].replace(
    ['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse'], 'married')

# Objects are mapped to numerical values
categorical_df = df.select_dtypes(include=['object'])

enc = LabelEncoder()

categorical_df = categorical_df.apply(enc.fit_transform)

df = df.drop(categorical_df.columns, axis=1)
df = pd.concat([df, categorical_df], axis=1)

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

clf = RandomForestClassifier(n_estimators=100, random_state=24)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Random Forests accuracy", accuracy_score(y_test, y_pred))
