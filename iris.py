# importing necessary libraries
import pandas as pd
import numpy as np
import pickle
df = pd.read_csv('Iris.csv')
# defining X and y
X = df.drop('Species', axis = 1)
y = df['Species']
# label encoding the target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# fitting the model
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# importing the algorithm
from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)
pickle.dump(sv, open('iris.pkl', 'wb'))
