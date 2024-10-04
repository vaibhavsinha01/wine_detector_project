from sklearn.datasets import load_wine
import pandas as pd
from sklearn.metrics import precision_score,accuracy_score,recall_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import numpy as np
import joblib

data=load_wine()
new_data = pd.DataFrame(data.data,columns=data.feature_names)
new_data['target'] = data.target

split = int(len(new_data)*0.8)

X = new_data.drop('target',axis=1)
y = new_data['target']

x_train = X.iloc[:split]
y_train = y.iloc[:split]
x_test = X.iloc[split:]
y_test = y.iloc[split:]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(128,input_dim=13,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=20,batch_size=20,validation_data=(x_test,y_test))

loss,accuracy = model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

score_acc = accuracy_score(y_test, y_pred_classes)
score_pre = precision_score(y_test, y_pred_classes, average='weighted')

print('The accuracy and precision scores are:')
print(score_acc)
print(score_pre)

# Save the model
model.save('wine_prediction_model_nn_70.h5')



