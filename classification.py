#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#data collection and preprocessing
#loading the data from csv file to pandas dataframe
raw_mail_data=pd.read_csv('mail_data.csv')
#print(raw_mail_data)
#it is observed that the data has many null values therefore we want to replace it with null strings
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')
#printing first five rows of dataframe
#print(mail_data.head())
#checking number of rows and colums in the dataframe
#print(mail_data.shape)
#now what we will do is label encoding which means encoding label to numerical values
#label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category']=='spam','Category']=0
mail_data.loc[mail_data['Category']=='ham','Category']=1
#spam=0,ham=1
#saperating the data as text and label
#creating 2 variables x and y where x is text and y is category
X= mail_data['Message']
Y= mail_data['Category']
#print(X)
#print(Y)
#splitting the data into training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
# print(X.shape)
# print(X_test.shape)
# print(X_train.shape)
#now we need to convert all this text data into meaningful numerical values
# for this we use feature extraction
#we have imported tfidfvectorizer to convert this text data
#transform text data to feature vectors that can be used as input to the logistic regression model
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
#what tfidf does is it goes through the  text and finds the frequency of each word and it gives them a score proportionately.
#it gives scores to every word in the dataset
#then they are linked to label 0 and 1
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)
#convert y_train and y_test as integers because we saw that the type is shown as object
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')
#print(X_train_features)
#now we will use logistic regression to train the model
#we have imported logistic regression from sklearn.linear_model
#we will use the fit method to train the model
model=LogisticRegression()
model.fit(X_train_features,Y_train)
#we  will use the predict method to predict the output
#we will use the accuracy_score function to check the accuracy of the model
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)
#print(accuracy_on_training_data)
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
#print(accuracy_on_test_data)
#building a predictive system
input_mail=["Even my brother does not like to speak with me. They treat me like aids patent."]
#convert text to feature vectors
input_mail_features=feature_extraction.transform(input_mail)
#making predictions
prediction=model.predict(input_mail_features)
#print(prediction)
if prediction[0]==1:
    print("ham mail")
else:
    print("spam mail")    





