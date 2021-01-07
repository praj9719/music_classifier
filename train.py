import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def train(data, col, model):
    y = data[col]
    x = data.drop([col], axis=1)

    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.33)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return accuracy_score(y_test, prediction)


dataset = pd.read_csv('data.csv')
dataset = dataset.drop(['filename'], axis=1)
dataset.label = pd.factorize(dataset.label)[0]
algo = SVC(kernel="linear", C=0.025, random_state=101)
y = 'label'
accur = train(dataset, y, algo)
accur = round(accur*100, 2)
print(accur)

