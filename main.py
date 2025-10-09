from sentiment_analyzer import SentimentAnalyzer
import streamlit as st
import nltk

st.set_page_config(page_title="Product Review Sentiment Analyzer", page_icon="💬")

st.title("Product Review Sentiment Analyzer")
st.write("Enter a product review below and click **Analyze** to see its sentiment.")

# Load models once
@st.cache_resource
def load_analyzer():
    nltk.download('punkt_tab')
    return SentimentAnalyzer(
        model_path="model/sentiment_analysis_model.h5",
        w2v_model_path="model/word2vec_model.bin",
        tokenizer_path="model/tokenizer.pickle",
        max_len=300,
        embedding_dim=300
    )

analyzer = load_analyzer()

# Input text box
review = st.text_area("Enter your review:", height=150)

# Analyze button
if st.button("Analyze Sentiment"):
    if review.strip():
        sentiment = analyzer.predict_sentiment(review)
        st.success(f"**Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a review first.")