import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import urllib
from PIL import Image

coltrans = ColumnTransformer([
      ('imputer-mod_onehot', Pipeline([
                                ('imputer-mod', SimpleImputer(strategy='most_frequent')),
                                ('onehot', OneHotEncoder(drop='first')),
                              ]), ['Gender', 'Married', 'Self_Employed']),
      ('imputer-mod', SimpleImputer(strategy='most_frequent'), ['Credit_History']),
      ('imputer-mean_scaling', Pipeline([
                                ('imputer-mean', SimpleImputer(strategy='mean')),
                                ('scaling', StandardScaler()),
                              ]), ['Dependents', 'LoanAmount', 'Loan_Amount_Term']),
      ('onehot', OneHotEncoder(drop='first'), ['Education', ]),
      ('scaling', StandardScaler(), ['ApplicantIncome', 'CoapplicantIncome', ]),
])

@st.cache_data
def get_data():
    X = pd.read_csv("https://raw.githubusercontent.com/Ramanand-Yadav/EligibilityForBankLoanMLProject/main/trainDataset.csv")
    X.drop('Loan_ID', axis = 1, inplace = True)

    Y = X['Loan_Status']
    X = X.drop('Loan_Status', axis=1)

    X['Dependents'].replace({'3+': 3}, inplace = True)
    X['Dependents'] = pd.to_numeric(X['Dependents'])
    Y = np.where(Y == 'Y', 1, 0)
    return X, Y

X, Y = get_data()
columns = X.columns
X = coltrans.fit_transform(X)

@st.cache_data
def get_model():
    model = LogisticRegression().fit(X, Y)
    return model

model = get_model()

@st.cache_data
def get_image():
    urllib.request.urlretrieve(
        'data:image/jpeg;base64,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',
        'img.jpg'
    )

get_image()

data = {}
img = Image.open('img.jpg')
st.image(img,width=500)

st.header('BLOAN --- Will you get the LOAN??')

data['Gender'] = st.selectbox('Gender', ['Male', 'Female'])
data['Married'] = st.selectbox('Married', ['Yes', 'No'])
data['Dependents'] = st.number_input('Dependents', min_value=0)
data['Education'] = st.selectbox('Education', ['Graduate', 'Not Graduate'])

map = {'Job': 'No', 'Bussiness': 'Yes'}
data['Self_Employed'] = st.selectbox('Self_Employed', ['Job', 'Bussiness']) # format_func
data['Self_Employed'] = map[data['Self_Employed']]

data['ApplicantIncome'] = st.number_input('ApplicantIncome', min_value=0)
data['CoapplicantIncome'] = st.number_input('CoapplicantIncome', min_value=0)
data['LoanAmount'] = st.number_input('LoanAmount', min_value=0)
data['Loan_Amount_Term'] = st.number_input('Loan_Amount_Term', min_value=0)
data['Property_Area'] = st.selectbox('Property_Area', ['Rural', 'Urban', 'Semiurban	'])

map = {'Good': 1, 'Average': 0}
data['Credit_History'] = st.selectbox('Transaction Frequency', ['Good', 'Average']) # format_func
data['Credit_History'] = map[data['Credit_History']]


features = pd.DataFrame(data, index = [0], columns = columns)
st.dataframe(features)



if st.button('Submit'):
    features = coltrans.transform(features)
    pred = model.predict(features)[0]

    if pred == 1:
        st.success('Loan will Approve')
    else:
        st.error('Loan will not Approve')
