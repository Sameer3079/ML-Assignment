import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder


# Convert Strings to Numeric Values
def format_data(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data


# Load Data
raw_data = pd.read_csv('./data/adult.data')
print("Data Set Shape =", raw_data.shape)

raw_data = format_data(raw_data)
sample_count = len(raw_data)
test_percentage = 0.1
print("Train : Test Ratio =", 1 - test_percentage, ":", test_percentage)
test_sample_count = test_percentage * sample_count

test_sample_count = test_sample_count.__round__()
train_sample_count = sample_count - test_sample_count

train_X = raw_data.iloc[test_sample_count:, :14]
train_Y = raw_data.iloc[test_sample_count:, 14:]
test_X = raw_data.iloc[:test_sample_count, :14]
test_Y = raw_data.iloc[:test_sample_count, 14:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)
print("Trained Classifier")

# Check Accuracy
print("Checking Accuracy")
correct_count = 0
for x in range(test_sample_count):
    prediction = clf.predict(test_X.iloc[[x]])
    if prediction[0] == test_Y.iat[x, 0]:
        correct_count += 1

accuracy = (correct_count / test_sample_count) * 100
print("Accuracy = ", str(accuracy.__round__()) + "%")
