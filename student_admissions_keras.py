import pandas as pd
import numpy as np
import keras, math
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adagrad, rmsprop

###try relu and sigmoid
###try categorical_crossentropy, mean_squared_error,
###try rmsprop, adam, ada

#load dataset
df=pd.read_csv('/Users/dylanrutter/Downloads/aind2-dl-master/student_data.csv')
df=df.fillna(0) #fill NA slots with 0
df = pd.get_dummies(df, columns=['rank'])

df['gre'] = df['gre']/800 #normalize from range 0 to 800
df['gpa'] = df['gpa']/4

X = np.array(df)[:,1:]#makes array where each element is a vector of row
X = X.astype('float32')

#makes labels into one-hot array
y = df['admit'].values.ravel()
num_classes = np.max(y) + 1
n = y.shape[0]
categorical = np.zeros((n, num_classes))
categorical[np.arange(n),y] = 1
y = categorical

#splits data into train, valid, and test sets of features and labels
X_train, X_valid, X_test = X[:900], X[900:1050], X[1050:1200]
y_train, y_valid, y_test = y[:900], y[900:1050], y[1050:1200]

#set up univeral variables
num_features = X_train.shape[1]
num_labels = y_train.shape[1]

# Building the model
# Note that filling out the empty rank as "0", gave us an extra column,
# for "Rank 0" students.
# Thus, our input dimension is 7 instead of 6.
model = keras.models.Sequential()
model.add(Dense(128, activation='relu', input_shape=(7,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam',\
              metrics=['accuracy'])
#model.summary()
model.fit(X_train, y_train, nb_epoch=200, batch_size=100, verbose=0)

score = model.evaluate(X_train, y_train)
print("\n Training Accuracy:", score[1])
score = model.evaluate(X_test, y_test)
print("\n Testing Accuracy:", score[1])
