#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

url = "https://finance.yahoo.com/topic/economic-news/"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find the news articles
articles = soup.find_all("h3")

# Extract and display the news titles
for article in articles:
    print(article.text.strip())
    print("------")

# Check for pagination
pagination = soup.find("a", {"class": "next"})

# If pagination exists, navigate to the next page
while pagination:
    next_page_url = "https://finance.yahoo.com" + pagination["href"]
    response = requests.get(next_page_url)
    soup = BeautifulSoup(response.content, "html.parser")
    articles = soup.find_all("h3")
    
    for article in articles:
        print(article.text.strip())
        print("------")
    
    pagination = soup.find("a", {"class": "next"})

# Sample news articles (replace these with actual scraped news articles)
news_articles = [
    "The company reported record-breaking earnings this quarter.",
    "New regulations could affect the tech industry.",
    "Stock market reacts to interest rate hike announcement."
]

# Keywords to filter relevant news
keywords = ["earnings", "regulation", "stock market"]

# Analyze sentiment and filter relevant news
relevant_news = []
for article in news_articles:
    for word in keywords:
        if word in article.lower():
            sentiment = TextBlob(article).sentiment.polarity
            if sentiment != 0:  # Filter neutral sentiment
                relevant_news.append((article, sentiment))
                break

# Display relevant news articles with sentiment
for news, sentiment in relevant_news:
    print(f"News: {news} | Sentiment: {sentiment}")

