import requests, json, os
from textblob import TextBlob
from textblob.en.sentiments import PatternAnalyzer
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# get API from secret environment variable
API_KEY = os.getenv('API_KEY')

def fetch_reviews(api_key, location, term):
    url = f"https://api.yelp.com/v3/businesses/search?location={location}&term={term}" 
    headers = {'Authorization': f'Bearer {api_key}',
               'accept': 'application/json'}
    response = requests.get(url, headers=headers)
    businesses = response.json()['businesses']
    reviews = []
    num_businesses = 60
    for business in businesses:
        business_id = business['alias']
        url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews?limit=20&sort_by=yelp_sort"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            continue
        reviews += response.json()['reviews']
        num_businesses -= 1
        if num_businesses == 0:
            break
    return reviews


def clean_reviews(reviews):
    stop_words = set(stopwords.words('english'))
    cleaned_reviews = []
    for review in reviews:
        text = review['text'].lower()
        words = text.split()
        cleaned_words = [word for word in words if word not in stop_words]
        cleaned_reviews.append(' '.join(cleaned_words))
    return cleaned_reviews


def analyze_sentiment(reviews):
    textblob_sentiments = []
    pattern_sentiments = []
    for review in reviews:
        textblob_sentiment = TextBlob(review).sentiment.polarity
        textblob_sentiments.append(textblob_sentiment)
        
        pattern_sentiment = TextBlob(review, analyzer=PatternAnalyzer()).sentiment.polarity
        pattern_sentiments.append(pattern_sentiment)
    
    return textblob_sentiments, pattern_sentiments


def visualize_results(sentiments):
    # Assuming 'sentiments' is a list of sentiment scores
    # Convert to numpy array for easier manipulation
    sentiments_np = np.array(sentiments)
    
    # Calculate the total number of reviews
    total_reviews = len(sentiments_np)
    
    # Check for division by zero
    if total_reviews == 0:
        print("Error: No reviews to analyze.")
        return
    
    # Calculate the number of positive, neutral, and negative reviews
    positive_reviews = np.sum(sentiments_np > 0)
    neutral_reviews = np.sum(sentiments_np == 0)
    negative_reviews = np.sum(sentiments_np < 0)
    
    # Calculate the percentages
    positive_percentage = (positive_reviews / total_reviews) * 100
    neutral_percentage = (neutral_reviews / total_reviews) * 100
    negative_percentage = (negative_reviews / total_reviews) * 100
    
    # Ensure no division by zero and handle NaN values
    if np.isnan(positive_percentage) or np.isnan(neutral_percentage) or np.isnan(negative_percentage):
        print("Error: Division by zero or NaN values encountered.")
        return
    
    # Prepare data for the pie chart
    sizes = [positive_percentage, neutral_percentage, negative_percentage]
    labels = ['Positive', 'Neutral', 'Negative']
    
    # Create the pie chart
    fig, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.show()


def create_wordcloud(cleaned_reviews):
    """
    Creates a word cloud of the 20 most common words from the cleaned reviews.
    """
    # Combine all cleaned reviews into a single string
    combined_reviews = ' '.join(cleaned_reviews)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=20, background_color='white').generate(combined_reviews)
    
    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Fetch reviews
    reviews = fetch_reviews(API_KEY, "Roxbury NJ", "diners")

    # Analyze sentiment
    cleaned_reviews = clean_reviews(reviews)
    textblob_sentiments, naivebayes_sentiments = analyze_sentiment(cleaned_reviews)
    
    # Visualize results
    visualize_results(textblob_sentiments)
    create_wordcloud(cleaned_reviews)