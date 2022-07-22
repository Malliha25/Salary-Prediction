import streamlit as st
st.header('Salary prediction System')
import pandas as pd
dataset=pd.read_csv('Salary_Data_SLR.csv')
X=dataset.iloc[:,0].values
Y=dataset.iloc[:,1].values
X=X.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_pred,Y_test))
p=st.text_input("years of experience")
if st.button('predict salary'):
    salary=regressor.predict([[p]])[0]
    salary=int(salary)
    st.success(salary)