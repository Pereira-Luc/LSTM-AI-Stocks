# Importing necessary libraries
from pygooglenews import GoogleNews
from datetime import datetime, timedelta
import csv

# Function to get titles of news articles
def get_titles(search, from_date, to_date):
    stories = []
    gn = GoogleNews()  # Creating an instance of GoogleNews
    search_query = f'intitle:{search}'  # Formatting the search query to include the title
    search = gn.search(search_query, from_=from_date, to_=to_date)  # Performing the search
    newsitem = search['entries']  # Extracting the news items from the search results

    # Iterating over each news item and storing the required details
    for item in newsitem:
        story = {
            'title': item.title,  # Title of the news article
            'summary': item.summary,  # Summary of the news article
            'link': item.link  # URL of the news article
        }
        stories.append(story)  # Adding the story details to the list
    return stories  # Returning the list of stories

# Function to save data to a CSV file
def save_to_csv(filename, data):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:  # Opening the CSV file in append mode
        writer = csv.writer(file)  # Creating a CSV writer object
        for row in data:
            writer.writerow(row)  # Writing each row of data to the CSV file

# List of topics to search
topics = ['Amazon Corporation', 'Tesla Inc.', 'Apple Inc', 'Microsoft Corporation', 'Netflix']

# Setting the start and end dates for the search
start_date = datetime(2007, 1, 1)
end_date = datetime.today()

# Iterating over each topic
for topic in topics:
    filename = f"{topic}.csv"  # Creating a filename for each topic
    # Write headers to the CSV file
    save_to_csv(filename, [("Day", "Title")])  # Writing headers to the CSV file

    # Iterating over each day within the date range
    for single_date in (start_date + timedelta(n) for n in range((end_date - start_date).days)):
        from_date = single_date.strftime("%Y-%m-%d")  # Formatting the start date
        to_date = (single_date + timedelta(days=1)).strftime("%Y-%m-%d")  # Formatting the end date
        stories = get_titles(topic, from_date, to_date)  # Getting stories for the topic and date range

        # Writing each story's title to the CSV file
        for story in stories:
            save_to_csv(filename, [(from_date, story['title'])])  # Saving the title and date to the CSV
