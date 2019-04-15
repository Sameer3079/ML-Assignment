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

X = raw_data.iloc[:, :14]
Y = raw_data.iloc[:, 14:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print("Trained Classifier\n")

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
