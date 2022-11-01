import pandas as pd
import streamlit as st
import numpy as np
import pickle as p

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#dataset
data = pd.read_csv("num.csv")
data_sample = pd.read_csv("importance.csv")

#Text
st.title('URL Classification Prediction Model')
st.header(
          'This project aims to provide a comprehensive survey and a structural understanding of Malicious URL Detection techniques using machine learning. ')

st.text("Kindly fill out all the input fields in order to see the results")

st.dataframe(data_sample)

path_token_count = st.number_input("Path Token Count")
average_domain_token = st.number_input("Average Domain Token")
entropy_url = st.number_input("Entropy URL")
charcompvowels = st.number_input("Charcomp Vowels")
charcompace = st.number_input("Charcomp Ace")
path_url_ratio = st.number_input("Path Url Ratio")
domain_url_ratio = st.number_input("Domain URL Ratio")
symbol_count_url = st.number_input("Symbol Count URL")

features = ['path_token_count', 'average_domain_token', 'entropy_url', 'charcompvowels', 'charcompace', 'path_url_ratio', 'domain_url_ratio', 'symbol_count_url']

# Labeling X and y features
X = data[features]
y = data['url_type']
#X = np.nan_to_num(X)

# Training the data using decision tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
dtc_clf = DecisionTreeClassifier()
dtc_clf.fit(X_train,y_train)
predict_val = dtc_clf.predict([[path_token_count, average_domain_token, entropy_url, charcompvowels, charcompace, path_url_ratio, domain_url_ratio, symbol_count_url]])
predict_val = float(predict_val)

if predict_val == 1:
    st.info("URL Type: Benign")

elif predict_val == 0:
    st.info("URL Type: Defacement")

elif predict_val == 2:
    st.info("URL Type: Malware")

elif predict_val == 3:
    st.info("URL Type: Phishing")

else:
    st.info("URL Type: Spam")