# Pgming-methods

split test and train data
from sklearn.model_selection import train_test_split X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

Linear model
from sklearn.linear_model import LinearRegression regression=LinearRegression() regression.fit(X_train,y_train)

#
1
#y_pred=regression.predict(X_test)
find rscore or precision are same
TP/TP+FN from sklearn.metrics import r2_score score=r2_score(y_test,y_pred)

Accuracy
TP+TN/TP+TN+FN+FP from sklearn.metrics import accuracy_score accuracy = accuracy_score(y_test, y_pred)

STANDARD SCALAR
from sklearn.preprocessing import StandardScaler sc_X = StandardScaler() X_train = sc_X.fit_transform(X_train) X_test = sc_X.transform(X_test) sc_y = StandardScaler() y_train = sc_y.fit_transform(y_train)

MIN-MAX SCALAR
from sklearn.preprocessing import MinMaxScaler scaler = MinMaxScaler() scaler.fit(X) X_scaled = scaler.transform(X)

lOGISTIC ML
from sklearn.linear_model import LinearRegression regression=LinearRegression() regression.fit(X_train,y_train)

CONFUSION MATRIX
from sklearn.metrics import confusion_matrix cf = confusion_matrix(y_test, y_pred) plt.figure() sns.heatmap(cf, annot=True) plt.xlabel('Prediction') plt.ylabel('Target') plt.title('Confusion Matrix')
