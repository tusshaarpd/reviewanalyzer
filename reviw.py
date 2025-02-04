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
from urllib.parse import urlparse

class ReviewScraper:
    def __init__(self):
        self.ua = UserAgent()
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }

    def find_review_elements(self, soup):
        """
        Smart detection of review elements using common patterns
        """
        review_patterns = {
            'class_patterns': [
                r'review',
                r'comment',
                r'feedback',
                r'testimonial',
                r'rating',
                r'customer.*response'
            ],
            'text_patterns': [
                r'stars?',
                r'rated',
                r'rating',
                r'review',
                r'feedback'
            ]
        }
        
        potential_reviews = []
        
        # Search by class patterns
        for pattern in review_patterns['class_patterns']:
            elements = soup.find_all(class_=re.compile(pattern, re.I))
            potential_reviews.extend(elements)
        
        # Search by text content
        for pattern in review_patterns['text_patterns']:
            elements = soup.find_all(text=re.compile(pattern, re.I))
            potential_reviews.extend([elem.parent for elem in elements if elem.parent])
        
        # Search common container elements
        container_tags = ['article', 'div', 'section', 'li']
        for tag in container_tags:
            elements = soup.find_all(tag, class_=True)
            potential_reviews.extend([elem for elem in elements if any(
                re.search(pattern, str(elem.get('class', '')), re.I) 
                for pattern in review_patterns['class_patterns']
            )])
        
        return list(set(potential_reviews))

    def extract_text(self, element):
        """
        Extract review text from an element
        """
        # Try to find paragraph or text content
        text_elements = element.find_all(['p', 'span', 'div'], class_=re.compile(r'text|content|description', re.I))
        if not text_elements:
            text_elements = [element]
        
        texts = []
        for elem in text_elements:
            text = elem.get_text(strip=True)
            if len(text) > 20:  # Minimum length to be considered a review
                texts.append(text)
        
        return ' '.join(texts) if texts else None

    def extract_rating(self, element):
        """
        Extract rating from various formats
        """
        rating = None
        
        # Look for common rating patterns
        rating_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:star|★|out of \d+)', lambda x: float(x)),
            (r'rated\s*(\d+(?:\.\d+)?)', lambda x: float(x)),
            (r'rating:\s*(\d+(?:\.\d+)?)', lambda x: float(x))
        ]
        
        element_text = element.get_text()
        for pattern, converter in rating_patterns:
            match = re.search(pattern, element_text, re.I)
            if match:
                try:
                    rating = converter(match.group(1))
                    break
                except:
                    continue
        
        # Look for rating in attributes
        if not rating:
            for attr in ['data-rating', 'data-score', 'data-stars']:
                value = element.get(attr)
                if value:
                    try:
                        rating = float(value)
                        break
                    except:
                        continue
        
        return rating

    def scrape_reviews(self, url, num_pages=3):
        """
        Generic review scraper for any website
        """
        all_reviews = []
        base_url = url
        
        try:
            for page in range(1, num_pages + 1):
                # Handle pagination
                current_url = base_url
                if page > 1:
                    # Try common pagination patterns
                    if '?' in base_url:
                        current_url = f"{base_url}&page={page}"
                    else:
                        current_url = f"{base_url}?page={page}"
                
                headers = self.get_headers()
                try:
                    response = requests.get(current_url, headers=headers, timeout=15)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'lxml')
                    
                    # Find potential review elements
                    review_elements = self.find_review_elements(soup)
                    
                    if not review_elements:
                        st.warning(f"No reviews found on page {page}. Moving to next page...")
                        break
                    
                    for element in review_elements:
                        try:
                            # Extract review text
                            review_text = self.extract_text(element)
                            if not review_text:
                                continue
                            
                            # Extract rating
                            rating = self.extract_rating(element)
                            
                            # Extract date if available
                            date_element = element.find(class_=re.compile(r'date|time|posted', re.I))
                            review_date = date_element.get_text(strip=True) if date_element else None
                            
                            # Create review object
                            review_data = {
                                'text': review_text,
                                'rating': rating if rating else None,
                                'date': review_date
                            }
                            
                            all_reviews.append(review_data)
                        except Exception as e:
                            continue
                    
                    # Progress indicator
                    st.progress(page / num_pages)
                    
                    # Random delay between requests
                    time.sleep(random.uniform(2, 4))
                    
                except requests.RequestException as e:
                    st.error(f"Error fetching page {page}: {str(e)}")
                    break
            
            return all_reviews
        
        except Exception as e:
            st.error(f"Error in review collection: {str(e)}")
            return []

def analyze_sentiment(text):
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    except Exception as e:
        return 0, 0

def get_key_phrases(reviews):
    try:
        all_text = ' '.join([r['text'] for r in reviews if r['text']])
        blob = TextBlob(all_text)
        phrases = blob.noun_phrases
        return Counter(phrases).most_common(5)
    except Exception as e:
        return []

def setup_nltk():
    try:
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
        return True
    except Exception as e:
        st.error(f"Error setting up NLTK: {str(e)}")
        return False

def main():
    st.title("Universal Review Analyzer")
    
    if not setup_nltk():
        st.stop()
    
    st.markdown("""
    ### Instructions:
    1. Enter the company name
    2. Paste the URL of the reviews page
    3. Select number of pages to analyze
    4. Click 'Analyze Reviews'
    """)
    
    company_name = st.text_input("Enter Company Name:")
    url = st.text_input("Enter Reviews URL:")
    num_pages = st.slider("Number of pages to analyze", min_value=1, max_value=10, value=3)
    
    if st.button("Analyze Reviews"):
        if url:
            with st.spinner(f"Analyzing reviews from {num_pages} pages... This may take a few moments..."):
                scraper = ReviewScraper()
                reviews = scraper.scrape_reviews(url, num_pages)
                
                if reviews:
                    # Calculate sentiments
                    for review in reviews:
                        if review['text']:
                            polarity, subjectivity = analyze_sentiment(review['text'])
                            review['sentiment_polarity'] = polarity
                            review['sentiment_subjectivity'] = subjectivity
                    
                    # Create DataFrame
                    df = pd.DataFrame(reviews)
                    
                    # Display results
                    st.header(f"Analysis Results for {company_name}")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_reviews = len(reviews)
                        st.metric("Total Reviews", total_reviews)
                    with col2:
                        positive_reviews = len(df[df['sentiment_polarity'] > 0])
                        st.metric("Positive Reviews", f"{positive_reviews}/{total_reviews}")
                    with col3:
                        negative_reviews = len(df[df['sentiment_polarity'] < 0])
                        st.metric("Negative Reviews", f"{negative_reviews}/{total_reviews}")
                    
                    # Rating stats if available
                    if 'rating' in df.columns and df['rating'].notna().any():
                        st.metric("Average Rating", f"{df['rating'].mean():.1f}⭐")
                        
                        # Rating distribution
                        fig2 = px.histogram(df, x="rating",
                                          title="Rating Distribution",
                                          labels={'rating': 'Rating'},
                                          nbins=5,
                                          color_discrete_sequence=['#3366cc'])
                        st.plotly_chart(fig2)
                    
                    # Sentiment distribution plot
                    fig = px.histogram(df, x="sentiment_polarity",
                                     title="Sentiment Distribution",
                                     labels={'sentiment_polarity': 'Sentiment Score'},
                                     color_discrete_sequence=['#3366cc'])
                    st.plotly_chart(fig)
                    
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
                        most_positive = max(reviews, key=lambda x: x.get('sentiment_polarity', 0))
                        review_text = most_positive.get('text', 'No text available')
                        rating_text = f"Rating: {most_positive['rating']}⭐" if most_positive.get('rating') else ''
                        st.info(f"{rating_text}\n\n{review_text}")
                    with col2:
                        st.write("Most Negative Review:")
                        most_negative = min(reviews, key=lambda x: x.get('sentiment_polarity', 0))
                        review_text = most_negative.get('text', 'No text available')
                        rating_text = f"Rating: {most_negative['rating']}⭐" if most_negative.get('rating') else ''
                        st.error(f"{rating_text}\n\n{review_text}")
                    
                    # Show all reviews in a table
                    st.subheader("All Reviews")
                    display_df = df.copy()
                    for col in ['text', 'rating', 'sentiment_polarity', 'date']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].fillna('N/A')
                    
                    st.dataframe(
                        display_df[['text', 'rating', 'sentiment_polarity', 'date']].rename(columns={
                            'text': 'Review',
                            'rating': 'Rating',
                            'sentiment_polarity': 'Sentiment',
                            'date': 'Date'
                        })
                    )
                    
                    # Export option
                    if st.button("Export Reviews"):
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{company_name}_reviews.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("No reviews found. Please check the URL and try again.")
        else:
            st.warning("Please enter a valid URL")

if __name__ == "__main__":
    main()
