import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

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

def preprocess_labels(labels):
    
    labels = labels.str.lower().str.strip()
    return labels


df = pd.read_csv(r"C:\Users\ACER\Desktop\assess\train.csv")


df.dropna(subset=['Tweets'], inplace=True)

df['preprocessed_tweet'] = df['Tweets'].apply(preprocess_text)

df['preprocessed_labels'] = preprocess_labels(df['label'])

print(df.head())


