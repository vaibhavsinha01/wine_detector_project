from sklearn.datasets import load_wine
import pandas as pd
from sklearn.metrics import precision_score,accuracy_score,recall_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

data=load_wine()
print(data)
new_data = pd.DataFrame(data.data,columns=['alcohol','malic_acid','ash','alcanity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline'])
new_data['target'] = data.target

split = int(len(new_data)*0.8)

X = new_data[['alcohol','malic_acid','ash','alcanity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline']]
y = new_data['target']

scaler = StandardScaler()
scaler.fit_transform(X)

x_train = X.iloc[:split]
y_train = y.iloc[:split]
x_test = X.iloc[split:]
y_test = y.iloc[split:]

reg = RandomForestClassifier()

reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)

acc_score = accuracy_score(y_pred,y_test)
pre_score = precision_score(y_pred,y_test,average='micro')
m = confusion_matrix(y_pred,y_test)

print(acc_score)
print(pre_score)
print(m)

joblib.dump(reg,'wine_detector.joblib')
