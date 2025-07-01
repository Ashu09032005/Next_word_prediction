# %%

import numpy as np
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import pandas as pd
## load the dataset
data=gutenberg.raw('shakespeare-hamlet.txt')
with open('hamlet.txt', 'w') as file:
    file.write(data)



import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
## load dataset
with open('hamlet.txt','r') as file:
    text = file.read().lower()
## Tokenize the etst creating indx for words
tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])








# %%
total_words=len(tokenizer.word_index) + 1
total_words

# %%
tokenizer.word_index

# %% [markdown]
# Input	Label (Next word)
# 
# "long	"live"
# 
# "long", "live"	"the"
# 
# "long", "live", "the"	"king"

# %%
## create input seq
inputsequences=[]
for line in text.split('\n'):
    ## convers each line to list of token ids
    token_list=tokenizer.texts_to_sequences([line])[0]
    ## iterate over length of line 
    
    for i in range(1, len(token_list)):
        n_gram_sequence=token_list[:i+1]
        inputsequences.append(n_gram_sequence)

# %%
inputsequences

# %%
## Pad seq
max_sequence_len=max([len(x) for x in inputsequences])
max_sequence_len


# %%
input_sequences=np.array(pad_sequences(inputsequences, maxlen=max_sequence_len, padding='pre'))

# %%
input_sequences

# %%
## create predictors and label
import tensorflow as tf
x,y=input_sequences[:,:-1],input_sequences[:,-1]


# %%
x

# %%
y

# %%
y=tf.keras.utils.to_categorical(y, num_classes=total_words)
## so wherever y index is present that will be 1 rest will be zero
y

# %%
#Split the data into train n test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
from tensorflow.keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# %%
## Train LSTM RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, max_sequence_len))
model.summary()

# %%
history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),verbose=1)

# %%
def predict_next_word(model,tokenizer,text,max_sequence_len):
   
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list)>=max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
       
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None

# %%
input_text="This movie was boring and"
print(f"Input text: {input_text}")
## len of ur input model
max_sequence_len=model.input_shape[1]
next_word= predict_next_word(model, tokenizer, input_text, max_sequence_len)
print(f"Next word prediction: {next_word}")



