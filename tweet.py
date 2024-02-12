import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


df= pd.read_csv(r"C:\Users\ACER\Desktop\assess\preprocessed_dataset_train.csv")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['preprocessed_tweet'])
sequences = tokenizer.texts_to_sequences(df['preprocessed_tweet'])

max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


labels = df['preprocessed_labels']
label_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
num_classes = len(label_dict)
labels_onehot = np.zeros((len(labels), num_classes))
for i, label in enumerate(labels):
    labels_onehot[i, label_dict[label]] = 1

embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(padded_sequences, labels_onehot, epochs=10, batch_size=32, validation_split=0.2)



train_loss, train_accuracy = model.evaluate(padded_sequences, labels_onehot)
print("Training Loss:", train_loss)
print("Training Accuracy:", train_accuracy)


model.save("lstm_train_model.h5")
