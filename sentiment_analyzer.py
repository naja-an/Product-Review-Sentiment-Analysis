import re
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

class SentimentAnalyzer:
    def __init__(self, model_path, tokenizer_path, w2v_model_path, max_len, embedding_dim):
        """
        Initialize Sentiment Analyzer.
        
        Args:
            model_path (str): Path to trained LSTM model (.h5)
            w2v_model_path (str): Path to trained Word2Vec model (.model)
            tokenizer_path (str): Path to saved tokenizer (.pkl) 
            max_len (int): Maximum sequence length used during training
            embedding_dim (int): Word2Vec embedding dimension
        """
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Load LSTM model
        print("Loading LSTM model...")
        self.model = load_model(model_path)

        # Load Word2Vec model
        print("Loading Word2Vec model...")
        self.w2v_model = Word2Vec.load(w2v_model_path)

        # Load tokenizer (optional)
        if tokenizer_path:
            print("Loading tokenizer...")
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:
            self.tokenizer = None

        print("Models loaded successfully!")

    def clean_text(self, text):
        """Clean text (lowercase, remove punctuation, etc.)
        Args:
            text (str): Raw text
        """
        text = re.sub(r'[^a-zA-Z ]', '', text.lower())
        return text
        


    def predict_sentiment(self, text):
        """ Predict and return "Positive" or "Negative".

        Args:
            text (str): Text for sentiment prediction.
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        # Tokenize text
        tokens = word_tokenize(cleaned_text)
        # Convert tokens to sequences
        sequence = self.tokenizer.texts_to_sequences([tokens])
        # Pad sequences
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')

        prediction = self.model.predict(padded_sequence)
        print(prediction)
        if prediction[0][0] >0.5:
            return "Positive"
        else:
            return "Negative"
