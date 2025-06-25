import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #This function helps you split your dataset into training and testing subsets
from sklearn.metrics import accuracy_score #This function helps evaluate how well your model is performing:

import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#create model
model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
# Evaluate the model
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()