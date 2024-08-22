import nltk
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import string # to process standard python strings

f = open('symptom.txt','r', errors = 'ignore')

raw = f.read()
raw = raw.lower()# converts to lowercase

# nltk.download('punkt') # first time use only
# nltk.download('wordnet') # first time use only

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

sent_tokens[:2]

word_tokens[:5]

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# ord() - converts a character into its Unicode code value
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) # splits a given sentence into words

GREETING_INPUTS = ("hello", "hi","hiii","hii","hiiii","hiiii", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["Hi, are you suffering from any health issues?(Y/N)", "Hey, are you having any health issues?(Y/N)", "Hii there, are you having any health issues?(Y/N)", "Hi there, are you having any health issues?(Y/N)", "Hello, are you having any health issues?(Y/N)", "I am glad! You are talking to me, are you having any health issues?(Y/N)"]
Basic_Q = ("yes","y")
Basic_Ans = "okay,tell me about your symptoms"
Basic_Om = ("no","n")
Basic_AnsM = "thank you visit again"


# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Checking for Basic_Q
def basic(sentence):
    for word in Basic_Q:
        if sentence.lower() == word:
            return Basic_Ans

# Checking for Basic_QM
def basicM(sentence):
    for word in Basic_Om:
        if sentence.lower() == word:
            return Basic_AnsM
            

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens) #It is used on the training data so that we can scale the training data and also learn the scaling parameters
    
    vals = cosine_similarity(tfidf[-1], tfidf)
   
    idx = vals.argsort()[0][-2] #it returns an array of indices along the given axis of the same shape as the input array, in sorted order.
    flat = vals.flatten() #convert it to one dimension
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx] 
        return robo_response
    


def chat(user_response):
    user_response = user_response.lower()
    
    if(user_response != 'bye'):
        
        if(user_response == 'thanks' or user_response=='thank you'):
            flag=False
            #print("ROBO: You are welcome..")
            return "You are welcome.."
        
        elif(basicM(user_response) != None):
            return basicM(user_response)
        
        else:
            if(greeting(user_response) != None):
                #print("ROBO: "+greeting(user_response))
                return greeting(user_response)

            elif(basic(user_response) != None):
                return basic(user_response)

            else:
                #print("ROBO: ",end="")
                #print(response(user_response))
                return response(user_response)
                sent_tokens.remove(user_response)
                
    else:
        flag=False
        #print("ROBO: Bye! take care..")
        return "Bye! take care.."
        


