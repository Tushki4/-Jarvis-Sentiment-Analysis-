from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

#from twitterapistuff import *

ckey = "1dBmiSXjIg44zSIZJc1TH8hoE"
csecret = "e9nTVN5uXiUNBT8MbjJXIvn00EdhBd0lwn2MC9PUwEq64aLt1w"
atoken = "147900442-6w036u6LjbIzi7yAgaSvcSIYo6N6P96ToS8A1Mgi"
asecret = "QpKRcAEUZqXuassjQgvyE8i8xf7iV5BNbai5p4pRk7K8k"

class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)
        
        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

def on_error(self, status):
    print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["canada"])




        
