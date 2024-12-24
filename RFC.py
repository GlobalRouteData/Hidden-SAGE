import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


# .8508
# dataset = pd.read_csv('../train_data/tree_data.csv',header=None,names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s'])
#
# X = dataset.iloc[:,:-1].values
# y = dataset.iloc[:,18].values
dataset = pd.read_csv('../train_data/tree_data9.csv',header=None,names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q'])
print(np.isnan(dataset).any())
dataset.dropna(inplace=True)
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,16].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators=50, criterion='entropy',random_state=42)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

result = confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(result)

result1 = classification_report(y_test,y_pred)
print("Classification Report:")
print(result1)

result2 = accuracy_score(y_test,y_pred)
print("Accuracy Score:")
print(result2)

model_name = "complex_rel.pkl"
with open(model_name,'wb') as f:
    pickle.dump(classifier,f)















