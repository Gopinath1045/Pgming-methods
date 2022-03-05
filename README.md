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

REPLECE TO INT VALUES
df['Fuel_Type']=df['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})

COVNVERT TO TEXT
# create a TF-IDF vectorizer object
tfidf_vectorizer=TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)

# fit the object with the training data tweets
tfidf_vectorizer.fit(df_train.tweet)

# transform the train and test data
df_train_idf = tfidf_vectorizer.transform(df_train.tweet)
df_test_idf  = tfidf_vectorizer.transform(df_test.tweet)

Label encoder to convert classification into multiass binary
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# code for lable encoder
df_final['Airline']=labelencoder.fit_transform(df_final['Airline'])
df_final['Source']=labelencoder.fit_transform(df_final['Source'])
df_final['Destination']=labelencoder.fit_transform(df_final['Destination'])
df_final['Additional_Info']=labelencoder.fit_transform(df_final['Additional_Info'])


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 10, 10]]))
