import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("next_word_lstm.h5")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        background-color: #fff0f5;
        color: black;
        border: 2px solid #ad1457;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #ad1457;
        color: white;
        border: none;
        padding: 8px 20px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Prediction function
def predict_next_word(model, tokenizer, input_text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit UI
st.title("🔮 Hamlet Next Word Prediction with LSTM")
input_text = st.text_input("Enter a sentence:", "Have you had quiet")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1]
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.markdown(f"<h3 style='color: green;'>Next word prediction: <b>{next_word}</b></h3>", unsafe_allow_html=True)
