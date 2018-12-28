import tweepy
import csv
import pandas as pd
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
with open('crawledtweets1.csv', mode='w', encoding='utf-8') as csvFile:
    #Use csv Writer
    csvWriter = csv.writer(csvFile,lineterminator = '\n')

    csvWriter.writerow(['Date', 'Text', 'Username', 'Location'])
    for tweet in tweepy.Cursor(api.search,q="buhari -filter:retweets -filter:links",
                               lang="en", tweet_mode='extended',
                               since="2018-12-22").items(500):
            csvWriter.writerow([tweet.created_at, tweet.full_text, tweet.user.screen_name, tweet.user.location])      
        #else:
        #print (tweet.created_at, tweet.text)
            #csvWriter.writerow([tweet.created_at, tweet.full_text.encode('utf-8'), tweet.user.screen_name])
