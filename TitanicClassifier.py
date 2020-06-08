import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# %matplotlib inline


#load data
data = pd.read_csv("train.csv")

"""#Preparing Data"""

#get rid of useless columns
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
data.head(5)

#remove null observations
data = data.dropna()
data.head(5)

#change male top 1 and female to 0
temp = []
for val in data["Sex"]:
  if "female" in val:
    temp.append(0)
  else:
    temp.append(1)
temp[:10]
data["Sex"] = np.array(temp)

temp[:10]
data["Sex"] = np.array(temp)

#Change embarked values to ints
temp2 = []
for val in data["Embarked"]:
  if "S" in val:
    temp2.append(0)
  elif "Q" in val:
    temp2.append(1)
  else:
    temp2.append(2)
data["Embarked"] = np.array(temp2)

#Split Data into training and testing
x = data.drop("Survived", axis=1)
y = data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42)

#Standardize Columns
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

"""#Neural Network"""

#Making and training the model
mlpc = MLPClassifier(hidden_layer_sizes=(7,7), max_iter=500)
mlpc.fit(x_train, y_train)

#Predicting
mlpc_pred = mlpc.predict(x_test)

print(classification_report(y_test, mlpc_pred))
print(confusion_matrix(y_test, mlpc_pred))