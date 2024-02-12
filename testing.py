import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


model = load_model('lstm_train_model.h5')


new_tweets_df = pd.read_csv(r"C:\Users\ACER\Desktop\assess\test.csv", header=None)

tweets = new_tweets_df.iloc[:, 0]
labels = new_tweets_df.iloc[:, 1]

def preprocess_text(text):
 
    text = text.lower()


    text = re.sub(r'http\S+', '', text)

   
    text = re.sub(r'@[A-Za-z0-9]+', '', text)

    
    text = text.encode('ascii', 'ignore').decode('ascii')

   
    text = re.sub(r'[^a-zA-Z\s]', '', text)

  
    tokens = word_tokenize(text)

   
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

 
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


preprocessed_tweets = tweets.apply(preprocess_text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_tweets)
sequences = tokenizer.texts_to_sequences(preprocessed_tweets)
max_sequence_length = 36  
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


predictions = model.predict(padded_sequences)
predicted_labels = np.argmax(predictions, axis=1)


sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
predicted_sentiments = [sentiment_map[label] for label in predicted_labels]


predicted_sentiments = [sentiment.capitalize() for sentiment in predicted_sentiments]


correct_predictions = (labels == predicted_sentiments).sum()
total_predictions = len(labels)
accuracy = correct_predictions / total_predictions

print("Total Predictions:", total_predictions)
print("Correct Predictions:", correct_predictions)
print("Testing Accuracy:", accuracy)
