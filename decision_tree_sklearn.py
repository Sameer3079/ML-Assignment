import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder


def format_data(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data


raw_data = pd.read_csv('./data/adult.data')
print("Data Set Shape =", raw_data.shape)

raw_data = format_data(raw_data)
sample_count = len(raw_data)
print("No. of Samples = ", sample_count)
test_percentage = 0.1
test_sample_count = test_percentage * sample_count

test_sample_count = test_sample_count.__round__()
train_sample_count = sample_count - test_sample_count
print("test =", test_sample_count)
print("train =", train_sample_count)

print("Splitting Data")
train_X = raw_data.iloc[test_sample_count:, :14]
train_Y = raw_data.iloc[test_sample_count:, 14:]
test_X = raw_data.iloc[:test_sample_count, :14]
test_Y = raw_data.iloc[:test_sample_count, 14:]

print("train_X = ", train_X.shape)
print("train_Y = ", train_Y.shape)
print("test_X = ", test_X.shape)
print("test_Y = ", test_Y.shape)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_Y)
print("\nTrained Classifier\n")

# Predicting Y using X
print("Predicting Y using X")
print("[0] = Person earns less than or equal to 50K per year")
print("[1] = Person earns more than 50K per year\n")

prediction_data_1 = [[21, 'Private', 77516, 'Bachelors', 13, 'Never-married', 'Prof-specialty', 'Unmarried', 'Other',
                      'Male', 2174, 0, 40, 'India']]
prediction_data_2 = [
    [40, 'Private', 121772, 'Assoc-voc', 11, 'Married-civ-spouse', 'Craft-repair', 'Husband', 'Asian-Pac-Islander',
     'Male', 0, 0, 40, '?']]

prediction_data_frame = pd.DataFrame(prediction_data_2)
prediction_data_frame = format_data(prediction_data_frame)

prediction = clf.predict(prediction_data_frame)

print("Prediction =", prediction)

prediction_data_frame = pd.DataFrame(prediction_data_1)
prediction_data_frame = format_data(prediction_data_frame)

prediction = clf.predict(prediction_data_frame)

print("Prediction =", prediction)
