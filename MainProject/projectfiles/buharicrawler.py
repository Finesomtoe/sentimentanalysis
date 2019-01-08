import tweepy
import csv
import pandas as pd
import datetime

date = datetime.datetime.now().strftime("%Y%m%d")

####input your credentials here
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
with open('buharitweets'+date+'.csv', mode='w', encoding='utf-8') as csvFile:
    #Use csv Writer
    csvWriter = csv.writer(csvFile,lineterminator = '\n')

    csvWriter.writerow(['Date', 'Text', 'Username', 'Location'])
    for tweet in tweepy.Cursor(api.search,q="buhari -atiku",
                               lang="en", tweet_mode='extended', since='2019-01-07').items(5000):
        if 'retweeted_status' not in dir(tweet):
            if not tweet.entities['urls']:
                csvWriter.writerow([tweet.created_at, tweet.full_text, tweet.user.screen_name, tweet.user.location]) 
                