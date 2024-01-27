from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Required NLTK downloads
nltk.download('vader_lexicon')
nltk.download('punkt')
# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load a BERT model specifically for sentiment analysis
model = BertForSequenceClassification.from_pretrained('bert-base-uncased') # You need a model trained for sentiment analysis here
sia = SentimentIntensityAnalyzer()
# Function to calculate Type-Token Ratio (TTR)
def calculate_ttr(text):
    tokens = nltk.word_tokenize(text)
    if len(tokens) == 0: return 0
    return len(set(tokens)) / len(tokens)

file_names = ['Amazon', 'Apple', 'Tesla_Inc', 'Netflix', 'Microsoft_Corporation']
for file in file_names:
  file_path = 'path_to_csv/' + file + '.csv' # Adjust the path to match where your file is located
  df = pd.read_csv(file_path)   
  headlines = df['Title'].tolist()
  # Check if the DataFrame has the columns, if not, create them with default NaN values
  for column in ['sentiment_score', 'intensity_score', 'type_token_ratio']:
      if column not in df.columns:
          df[column] = pd.NA
  # Now run through the headlines and only calculate and update if the values are missing (i.e., NaN)
  for index, row in df.iterrows():
      if pd.isna(row['sentiment_score']) or pd.isna(row['intensity_score']) or pd.isna(row['type_token_ratio']):
          headline = row['Title']
          # Process with BERT, TextBlob, VADER, and calculate TTR
          # Add your BERT sentiment analysis code here
          # Tokenize the headline and prepare it for BERT
          inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True).to('cpu')
          with torch.no_grad():
              outputs = model(**inputs)
          # Apply softmax to the output logits to get probabilities
          probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
          # Get the highest probability as the sentiment score (you may need to adjust based on your label mapping)
          sentiment_score = probabilities[:, 1].item()
          intensity_score = sia.polarity_scores(headline)['compound']
          ttr = calculate_ttr(headline)
          # Update the DataFrame with the new values
          df.at[index, 'sentiment_score'] = sentiment_score
          df.at[index, 'intensity_score'] = intensity_score
          df.at[index, 'type_token_ratio'] = ttr
          # Save the modified DataFrame back to the CSV
  df.to_csv(file_path, index=False)
