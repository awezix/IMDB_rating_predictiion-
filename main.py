import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

# loading word index  of imdb
word_index=imdb.get_word_index()
reversed_index={value:key for key,value in word_index.items()}

# loading pretrained model
model=load_model('simple_rnn_imdb.h5')

# function to decode reviews
def decode_reviews(encoded_review):
    return ' '.join([reversed_index.get(i-3,'?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=pad_sequences([encoded_review],maxlen=500)
    return padded_review

'''# prediction funtion
def prediction(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] >0.4 else 'Negetive'
    return sentiment,prediction[0][0]
'''
# Streamlit app
st.title('IMDB movie review sentiment analysis')
st.write('Enter a movie review to classify as positive or negative')

# user input
user_input=st.text_area('MOvie Review')

if st.button('classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] >0.4 else 'Negetive'

    # Display result
    st.write(f'Sentiment :{sentiment}')
    st.write(f'Prediction score :{prediction[0][0]}')

else:
    st.write('Please enter a movie review')

