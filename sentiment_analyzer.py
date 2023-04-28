#import packages
import pandas as pd
import nltk
import os
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import json
from nltk.corpus import stopwords
import time
import math
import health_lexicon
from datetime import date
from sklearn.metrics import accuracy_score,classification_report
import datetime as dt


#nltk.download("all")
nltk.download("stopwords")
nltk.download("vader_lexicon")
nltk.download('punkt')

# Make a list of english stopwords
stopwords = nltk.corpus.stopwords.words("english")

# Extend the list with your own custom stopwords
my_stopwords = ['https', 'the', 'The', 'She', 'she']
stopwords.extend(my_stopwords)


#Module Functions:


def get_sentiment(text:str, analyser,desired_type:str='pos'):
    """ 
    Get sentiment from text
    """ 
    sentiment_score = analyser.polarity_scores(text)
    return sentiment_score[desired_type]

def normalize(score, alpha=10):
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score/math.sqrt((score*score) + alpha)
    return norm_score
    
def get_sentiment_scores(df,data_column):
    # update VADER lexicon with healtcare specific terms:
    #sentiment analyzer VADER in NLTK
    sid_analyzer = SentimentIntensityAnalyzer()

    health_lexicon = health_lexicon.health_lexicon

    sid_analyzer.lexicon.update(health_lexicon)

    df[f'{data_column} Positive Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'pos'))
    df[f'{data_column} Negative Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neg'))
    df[f'{data_column} Neutral Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'neu'))
    df[f'{data_column} Compound Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'compound'))
    df[f'{data_column} Sentences'] = df[data_column].astype(str).apply(lambda x: nltk.tokenize.sent_tokenize(x))
    df[f'{data_column} Sentence_num'] = df[f'{data_column} Sentences'].apply(lambda x: len(x))
    return df

def get_overall_sentiment(df, data_column):
    # update VADER lexicon with healtcare specific terms:
    #sentiment analyzer VADER in NLTK
    sid_analyzer = SentimentIntensityAnalyzer()

    health_lexicon = health_lexicon.health_lexicon

    sid_analyzer.lexicon.update(health_lexicon)

    df[f'{data_column} Compound Sentiment Score'] = df[data_column].astype(str).apply(lambda x: get_sentiment(x,sid_analyzer,'compound'))
    return df

def find_anomalies(data):
    """
    Function to Detection Outliers on one-dimentional datasets
    """
    #define a list to accumlate anomalies
    anomalies = []
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * 3
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    #print(lower_limit)
    #print(upper_limit)
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append('Out of Control')
        else:
            anomalies.append('In Control')
    return anomalies

def sent_dict(lst):
    
    sid_analyzer = SentimentIntensityAnalyzer()

    health_lexicon = health_lexicon.health_lexicon

    sid_analyzer.lexicon.update(health_lexicon)
    
    k=[]
    for i in lst:
        k.append(i)
        k.append(get_sentiment(i, sid_analyzer, 'compound'))
    res_dct = {k[i]: k[i + 1] for i in range(0, len(k), 2)}
    return res_dct
    #problem converting dict type to json with decimals --> needs further attention

def sent_json(lst):
    
    sid_analyzer = SentimentIntensityAnalyzer()

    health_lexicon = health_lexicon.health_lexicon

    sid_analyzer.lexicon.update(health_lexicon)

    k=[]
    for i in lst:
        k.append(i)
        k.append(get_sentiment(i, sid_analyzer, 'compound'))
    res_dct = {k[i]: k[i + 1] for i in range(0, len(k), 2)}
    res_json = json.dumps(res_dct)
    return res_json



def get_sentiment_analysis(df, data_column):
    """
    This function breaks the comment column into sentences, applies the NLTK VADER sentiment analyzer (with the
    customized health Lexicon) to each of the sentences to generate a sum of the overall scores, 
    which are then normalized. Output includes: word token lists (words used in comments filtered for stop words), sentence scores list, and  
    anomaly detection. 
    
    """
    
    regexp = RegexpTokenizer('\w+')
    
    df[f'{data_column}'] = df[f'{data_column}'].fillna("No comment")
    df['SENTENCE_COMP'] = df[data_column].astype(str).apply(lambda x: nltk.tokenize.sent_tokenize(x))
    df['SENTENCE_NUM'] = df['SENTENCE_COMP'].apply(lambda x: len(x))    
    #df['SENTENCE_COMP'] = df['SENTENCE_BK'].apply(lambda x: sent_json(x))
    df['SENTENCE_COMP'] = df['SENTENCE_COMP'].apply(lambda x: sent_dict(x)) 
    df['SUM_SCORES'] = df['SENTENCE_COMP'].apply(lambda x: sum(x.values()))
    df['OVERALL_SCORE'] = df['SUM_SCORES'].apply(lambda x: normalize(x))
    df['OVERALL_SENTIMENT'] = df['OVERALL_SCORE'].apply(lambda c: 'Positive' if c >=0 else 'Negative')
    df['SENTIMENT_IND'] = df['OVERALL_SENTIMENT'].apply(lambda x: 1 if x =='Positive' else 0)
    df['TEXT_WORDS']=df[f'{data_column}'].apply(regexp.tokenize)
    # Remove stopwords
    df['TEXT_WORDS'] = df['TEXT_WORDS'].apply(lambda x: [item for item in x if item not in stopwords])
    df['TEXT_WORDS'] = df['TEXT_WORDS'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
    df['VIOL_DSC'] = find_anomalies(df['OVERALL_SCORE'])
    df['VIOL_IND'] = df['VIOL_DSC'].apply(lambda x: 1 if x =='Out of Control' else 0)

    #df = df.drop(columns=['SENTENCE_DI', 'SENTENCE_BK'])
    return df


def pd_prep(df):
    """
    prepares the pandas dataframe for conversion to pyspark. 
    """
    
    #deals with the issues raised with dictionary and list types as output. Need to change 
    df['SENTENCE_COMP'] = 'Test version'
    #define column types:
    #df['PRVDR_ID'] = df['PRVDR_ID'].astype(int, errors='ignore')
    #df['ATTENDING_PHYSICIAN_NPI'] = df['ATTENDING_PHYSICIAN_NPI'].astype(int, errors='ignore')
    #df['BUSINESS_UNIT'] = df['BUSINESS_UNIT'].astype(int, errors='ignore')
    #df['DEPT_NO'] = df['DEPT_NO'].astype(int, errors='ignore')
    #df['DEPT_NO'] = df['DEPT_NO'].fillna(0)
    #df['DEPT_NO'] = df['DEPT_NO'].astype(int, errors='ignore')
    df['SUM_SCORES'] = round(df['SUM_SCORES']*100)
    df['OVERALL_SCORE'] = round(df['OVERALL_SCORE']*100)
    df['SENTIMENT_IND'] = df['OVERALL_SENTIMENT'].apply(lambda x: 1 if x =='Positive' else 0)
    #df['SENTENCE_COMP'] = df['SENTENCE_COMP'].astype(str, errors='ignore')
    df['TEXT_WORDS'] = df['TEXT_WORDS'].astype(str, errors='ignore')
    df['EDP_LOAD_DTS'] = date.today() #change to DELTA_LOAD_DTS
    #df['SUM_SCORES'] = pd.to_numeric(df['SUM_SCORES'], downcast='integer')
    #df['OVERALL_SCORE'] = pd.to_numeric(df['OVERALL_SCORE'], downcast ='integer')
    return df

def results_matrix(df, data_column1, data_column2, before):
    trial_name = date.today()
    time = [before]
       
    print(classification_report(df[data_column1],df[data_column2]))
    print(accuracy_score(df[data_column1],df[data_column2]))
    with open(f'eval/{trial_name}{time}.txt',"w") as f:

        a_score = (accuracy_score(df[data_column1],df[data_column2]))
        print(f'The Accuracy Score is: {a_score}', file=f)
        print("  ")
        print(" --------------------------------- ", file=f)
        print("  ")
        print(classification_report(df[data_column1],df[data_column2]), file=f)
        print("  ")
        print(" --------------------------------- ", file=f)

    
