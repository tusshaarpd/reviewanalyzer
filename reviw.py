import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def scrape_reviews(url):
    """Scrapes reviews from a given URL."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = [review.get_text(strip=True) for review in soup.find_all('p')]
        return reviews if reviews else ["No reviews found."]
    except requests.exceptions.RequestException as e:
        return [f"Error fetching reviews: {e}"]

def analyze_sentiment(reviews):
    """Analyzes sentiment of reviews using an open-source AI model."""
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    results = sentiment_pipeline(reviews)
    
    positives = [rev for rev, res in zip(reviews, results) if res['label'] == 'POSITIVE']
    negatives = [rev for rev, res in zip(reviews, results) if res['label'] == 'NEGATIVE']
    
    return positives, negatives

def main():
    st.set_page_config(page_title="Reviews Analyzer", layout="wide")
    st.title("Reviews Analyzer - AI-powered Sentiment Analysis")
    st.write("Enter URLs containing reviews about a company to get insights.")
    
    url = st.text_input("Enter Review URL:")
    
    if st.button("Analyze Reviews"):
        if url:
            with st.spinner("Fetching and analyzing reviews..."):
                reviews = scrape_reviews(url)
                positives, negatives = analyze_sentiment(reviews)
                
                st.subheader("Positive Reviews:")
                for review in positives:
                    st.success(review)
                
                st.subheader("Negative Reviews:")
                for review in negatives:
                    st.error(review)
        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    main()
