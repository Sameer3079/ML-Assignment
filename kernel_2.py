import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/adult.csv")

print("DF Types =", df.dtypes)

df.isnull().sum()

df.columns.isna()

df.isin(['?']).sum()

df = df.replace('?', np.NaN)
df.head()

df = df.dropna()
df.head()

df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
df.income.head()

numerical_df = df.select_dtypes(exclude=['object'])
numerical_df.columns

plt.hist(df['age'], edgecolor='black')
plt.title('Age Histogram')
plt.axvline(np.mean(df['age']), color='yellow', label='average age')
plt.legend()
# plt.show()

age50k = df[df['income'] == 1].age
agel50k = df[df['income'] == 0].age

fig, axs = plt.subplots(2, 1)

axs[0].hist(age50k, edgecolor='black')
axs[0].set_title('Distribution of Age for Income > 50K')

axs[1].hist(agel50k, edgecolor='black')
axs[1].set_title('Distribution of Age for Income <= 50K')
plt.tight_layout()

df['marital.status'].unique()

ax = sns.countplot(df['marital.status'], hue=df['income'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()

df['marital.status'] = df['marital.status'].replace(['Widowed', 'Divorced', 'Separated', 'Never-married'], 'single')

df['marital.status'] = df['marital.status'].replace(
    ['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse'], 'married')

categorical_df = df.select_dtypes(include=['object'])
categorical_df.columns

sns.countplot(df['marital.status'], hue=df['income'])

enc = LabelEncoder()

ax = sns.countplot(df['income'], hue=df['race'])
ax.set_title('')

categorical_df = categorical_df.apply(enc.fit_transform)
categorical_df.head()

df = df.drop(categorical_df.columns, axis=1)
df = pd.concat([df, categorical_df], axis=1)
df.head()

sns.catplot(data=df, x='education', y='hours.per.week', hue='income', kind='point')

sns.FacetGrid(data=df, hue='income', height=6).map(plt.scatter, 'age', 'hours.per.week').add_legend()

plt.figure(figsize=(15, 12))
# plt.show()
cor_map = df.corr()
sns.heatmap(cor_map, annot=True, fmt='.3f', cmap='YlGnBu')

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

clf = RandomForestClassifier(n_estimators=100, random_state=24)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Random Forests accuracy", accuracy_score(y_test, y_pred))

decision_tree = DecisionTreeClassifier(criterion='gini', random_state=21, max_depth=10)

decision_tree.fit(X_train, y_train)
tree_prediction = decision_tree.predict(X_test)

print("Decision Tree accuracy: ", accuracy_score(y_test, tree_prediction))
