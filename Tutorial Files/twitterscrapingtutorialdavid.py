# Import the necessary package to process data in JSON format
import json
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

#class to fetch stream of tweets from twitter
class FetchData():

    #init function. Doesn't do anything interesting
    def __init__(self):
        self.type = 'test'

    #function to fetch tweets with a *query* parameter
    def fetch_tweets_with_query(self, query):
        # Variables that contains the user credentials to access Twitter API 
        ACCESS_TOKEN = 'access token'
        ACCESS_SECRET = 'access secret'
        CONSUMER_KEY = 'consumer key'
        CONSUMER_SECRET = 'consumer secret'
       
        oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

        # Initiate the connection to Twitter Streaming API
        twitter_stream = TwitterStream(auth=oauth)

        # Get a sample of the public data that contain the query term following through Twitter
        iterator = twitter_stream.statuses.filter(track=query, language='en')

        # Print each tweet in the stream to the screen 
        # Here we set it to stop after getting 500 tweets. 
        # You don't have to set it to stop, but can continue running 
        # the Twitter API to collect data for days or even longer. 
        tweet_count = 500
        for tweet in iterator:
            #reduce count of tweet_count
            tweet_count -= 1
            # Twitter Python Tool wraps the data returned by Twitter 
            # as a TwitterDictResponse object.
            # We convert it back to the JSON format to print/score
            print (json.dumps(tweet))
      
            #stop the loop if tweet_count is 0 or less
            if tweet_count <= 0:
                break 

#initialize class
df = FetchData()
#perform function
df.fetch_tweets_with_query('saraki')


