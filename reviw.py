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
import time
import random
from fake_useragent import UserAgent
import re

def setup_nltk():
    try:
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        nltk.data.path.append(nltk_data_dir)
        
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

def get_random_headers():
    ua = UserAgent()
    return {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

def get_trustpilot_reviews(base_url, num_pages=3):
    """
    Scrape multiple pages of reviews from Trustpilot
    """
    all_reviews = []
    
    try:
        for page in range(1, num_pages + 1):
            url = f"{base_url}?page={page}"
            headers = get_random_headers()
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Find all review containers
                review_containers = soup.find_all('article', class_=lambda x: x and 'paper' in x and 'review' in x)
                
                if not review_containers:
                    st.warning(f"No reviews found on page {page}. Moving to next page...")
                    continue
                
                for container in review_containers:
                    try:
                        # Get review text
                        review_text = container.find('p', {'data-service-review-text-typography': True})
                        if not review_text:
                            review_text = container.find('p', class_=lambda x: x and 'typography' in x)
                        
                        # Get rating
                        rating_div = container.find('div', {'data-service-review-rating': True})
                        if rating_div:
                            rating = rating_div.get('data-service-review-rating', '0')
                        else:
                            # Alternative method to find rating
                            rating_img = container.find('img', alt=re.compile(r'rated \d out of 5'))
                            if rating_img:
                                rating = re.search(r'rated (\d)', rating_img['alt']).group(1)
                            else:
                                rating = '0'
                        
                        if review_text:
                            all_reviews.append({
                                'text': review_text.text.strip(),
                                'rating': float(rating)
                            })
                    
                    except Exception as e:
                        st.warning(f"Error processing a review: {str(e)}")
                        continue
                
                # Add a small delay between pages
                time.sleep(random.uniform(1, 3))
                
            except requests.RequestException as e:
                st.error(f"Error fetching page {page}: {str(e)}")
                continue
        
        return all_reviews
    
    except Exception as e:
        st.error(f"Error in review collection: {str(e)}")
        return []

def analyze_sentiment(text):
    """
    Analyze sentiment of given text using TextBlob with error handling
    """
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    except Exception as e:
        st.warning(f"Error in sentiment analysis: {str(e)}")
        return 0, 0

def get_key_phrases(reviews):
    """
    Extract key phrases from reviews with error handling
    """
    try:
        all_text = ' '.join([r['text'] for r in reviews])
        blob = TextBlob(all_text)
        phrases = blob.noun_phrases
        return Counter(phrases).most_common(5)
    except Exception as e:
        st.warning(f"Error extracting key phrases: {str(e)}")
        return []

def main():
    st.title("Review Analyzer")
    
    if not setup_nltk():
        st.stop()
    
    company_name = st.text_input("Enter Company Name:")
    url = st.text_input("Enter Trustpilot Reviews URL:")
    num_pages = st.slider("Number of pages to analyze", min_value=1, max_value=10, value=3)
    
    if st.button("Analyze Reviews"):
        if url and "trustpilot.com" in url:
            with st.spinner(f"Analyzing reviews from {num_pages} pages... This may take a few moments..."):
                reviews = get_trustpilot_reviews(url, num_pages)
                
                if reviews:
                    # Calculate sentiments
                    for review in reviews:
                        polarity, subjectivity = analyze_sentiment(review['text'])
                        review['sentiment_polarity'] = polarity
                        review['sentiment_subjectivity'] = subjectivity
                    
                    # Create DataFrame
                    df = pd.DataFrame(reviews)
                    
                    # Display results
                    st.header(f"Analysis Results for {company_name}")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_rating = df['rating'].mean()
                        st.metric("Average Rating", f"{avg_rating:.1f}⭐")
                    with col2:
                        total_reviews = len(reviews)
                        st.metric("Total Reviews", total_reviews)
                    with col3:
                        positive_reviews = len(df[df['sentiment_polarity'] > 0])
                        st.metric("Positive Reviews", f"{positive_reviews}/{total_reviews}")
                    with col4:
                        negative_reviews = len(df[df['sentiment_polarity'] < 0])
                        st.metric("Negative Reviews", f"{negative_reviews}/{total_reviews}")
                    
                    # Sentiment distribution plot
                    fig = px.histogram(df, x="sentiment_polarity",
                                     title="Sentiment Distribution",
                                     labels={'sentiment_polarity': 'Sentiment Score'},
                                     color_discrete_sequence=['#3366cc'])
                    st.plotly_chart(fig)
                    
                    # Rating distribution
                    fig2 = px.histogram(df, x="rating",
                                      title="Rating Distribution",
                                      labels={'rating': 'Star Rating'},
                                      nbins=5,
                                      color_discrete_sequence=['#3366cc'])
                    st.plotly_chart(fig2)
                    
                    # Key phrases
                    st.subheader("Common Themes")
                    key_phrases = get_key_phrases(reviews)
                    for phrase, count in key_phrases:
                        st.write(f"• {phrase}: {count} mentions")
                    
                    # Sample reviews
                    st.subheader("Sample Reviews")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Most Positive Review:")
                        most_positive = max(reviews, key=lambda x: x['sentiment_polarity'])
                        st.info(f"Rating: {most_positive['rating']}⭐\n\n{most_positive['text']}")
                    with col2:
                        st.write("Most Negative Review:")
                        most_negative = min(reviews, key=lambda x: x['sentiment_polarity'])
                        st.error(f"Rating: {most_negative['rating']}⭐\n\n{most_negative['text']}")
                    
                    # Show all reviews in a table
                    st.subheader("All Reviews")
                    st.dataframe(
                        df[['text', 'rating', 'sentiment_polarity']].rename(columns={
                            'text': 'Review',
                            'rating': 'Rating',
                            'sentiment_polarity': 'Sentiment'
                        })
                    )
                else:
                    st.error("No reviews found. This could be due to website changes or access restrictions.")
        else:
            st.warning("Please enter a valid Trustpilot URL")

if __name__ == "__main__":
    main()
