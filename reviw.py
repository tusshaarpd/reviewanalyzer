import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import plotly.express as px
import os

# Create a custom directory for NLTK data
def setup_nltk():
    try:
        # Create a directory in the user's home directory
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', download_dir=nltk_data_dir)
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
            return True
        except Exception as e:
            st.error(f"Error downloading NLTK data: {str(e)}")
            st.info("Please run this command manually before starting the application: python -m nltk.downloader punkt averaged_perceptron_tagger")
            return False
            
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        return False

def scrape_reviews(url):
    """
    Scrape reviews from the given URL.
    Note: This is a basic implementation - you'll need to modify based on the specific website structure
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # This is a placeholder - modify the selector based on the actual website
        reviews = soup.find_all('div', class_='review-text')
        return [review.text.strip() for review in reviews]
    except Exception as e:
        st.error(f"Error scraping reviews: {str(e)}")
        return []

def analyze_sentiment(text):
    """
    Analyze sentiment of given text using TextBlob
    """
    analysis = TextBlob(text)
    # Return polarity (-1 to 1) and subjectivity (0 to 1)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def get_key_phrases(reviews):
    """
    Extract key phrases from reviews
    """
    all_text = ' '.join(reviews)
    blob = TextBlob(all_text)
    
    # Extract noun phrases
    phrases = blob.noun_phrases
    # Get most common phrases
    return Counter(phrases).most_common(5)

def main():
    st.title("Review Analyzer")
    
    # Setup NLTK data
    if not setup_nltk():
        st.stop()
    
    # Add company logo/name input
    company_name = st.text_input("Enter Company Name:")
    
    # Add URL input
    url = st.text_input("Enter Reviews URL:")
    
    if st.button("Analyze Reviews"):
        if url:
            with st.spinner("Analyzing reviews..."):
                # Scrape reviews
                reviews = scrape_reviews(url)
                
                if reviews:
                    # Calculate overall metrics
                    sentiments = [analyze_sentiment(review) for review in reviews]
                    polarities = [s[0] for s in sentiments]
                    subjectivities = [s[1] for s in sentiments]
                    
                    # Create sentiment distribution
                    sentiment_df = pd.DataFrame({
                        'Polarity': polarities,
                        'Subjectivity': subjectivities
                    })
                    
                    # Display results
                    st.header(f"Analysis Results for {company_name}")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_polarity = sum(polarities) / len(polarities)
                        st.metric("Average Sentiment", f"{avg_polarity:.2f}")
                    with col2:
                        positive_reviews = sum(1 for p in polarities if p > 0)
                        st.metric("Positive Reviews", f"{positive_reviews}/{len(reviews)}")
                    with col3:
                        negative_reviews = sum(1 for p in polarities if p < 0)
                        st.metric("Negative Reviews", f"{negative_reviews}/{len(reviews)}")
                    
                    # Sentiment distribution plot
                    fig = px.histogram(sentiment_df, x="Polarity",
                                     title="Sentiment Distribution",
                                     labels={'Polarity': 'Sentiment Score'},
                                     color_discrete_sequence=['#3366cc'])
                    st.plotly_chart(fig)
                    
                    # Key phrases
                    st.subheader("Key Phrases")
                    key_phrases = get_key_phrases(reviews)
                    for phrase, count in key_phrases:
                        st.write(f"• {phrase}: {count} mentions")
                    
                    # Sample positive and negative reviews
                    st.subheader("Sample Reviews")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Most Positive Review:")
                        most_positive = max(enumerate(reviews), key=lambda x: analyze_sentiment(x[1])[0])
                        st.info(most_positive[1])
                    with col2:
                        st.write("Most Negative Review:")
                        most_negative = min(enumerate(reviews), key=lambda x: analyze_sentiment(x[1])[0])
                        st.error(most_negative[1])
                else:
                    st.error("No reviews found at the provided URL")
        else:
            st.warning("Please enter a valid URL")

if __name__ == "__main__":
    main()
