# Import the necessary package to process data in JSON format
import json
# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

class FetchData():


    def __init__(self):
        self.type = 'test'

    def fetch_tweets_with_query(self, query):
        # Variables that contains the user credentials to access Twitter API 
        ACCESS_TOKEN = '1036999904877531136-1dN8kpEgNWt2uJqSU2cCTppIn8YKnC'
        ACCESS_SECRET = 'deUKZNORxOyc7M4ibXUkcQdEKObzRHICzVG58CtqHOZQr'
        CONSUMER_KEY = 'K709NnBB1ZNxI0HASOkxb8vfI'
        CONSUMER_SECRET = '7pVoyn4fMnbgJP0QDPeGo2aj1719D1bOogBWfuAurjNNl8E4MT'

        oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

        # Initiate the connection to Twitter Streaming API
        twitter_stream = TwitterStream(auth=oauth)

        # Get a sample of the public data following through Twitter
        iterator = twitter_stream.statuses.filter(track=query, language='en')

        # Print each tweet in the stream to the screen 
        # Here we set it to stop after getting 1000 tweets. 
        # You don't have to set it to stop, but can continue running 
        # the Twitter API to collect data for days or even longer. 
        tweet_count = 50
        for tweet in iterator:
            tweet_count -= 1
            # Twitter Python Tool wraps the data returned by Twitter 
            # as a TwitterDictResponse object.
            # We convert it back to the JSON format to print/score
            print (json.dumps(tweet))
      
            if tweet_count <= 0:
                break 

df = FetchData()
df.fetch_tweets_with_query('buhari')


