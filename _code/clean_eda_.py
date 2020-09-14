# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
### Cleaning & EDA Notebook 
# %%
from collections import Counter
import plotly.express as px
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb # data visualization library  
import random
import plotly.graph_objects as go


%matplotlib inline

# Allows full width row text in column
pd.set_option('display.max_colwidth', -1)
# %%
df = pd.read_csv('/Users/rashidbaset/Code/twitter-sentiment-prediction/_data/raw-data/Tweets.csv')
# %% [markdown]

Key Terminology

| Category                       | Definition                                                       |
| ------------------------------ | ---------------------------------------------------------------- |
| 'tweet_id'                     | Negative/neutral/positive Airline sentiment confidence           |
| 'airline_sentiment'            | Sentiment of tweet (target variable)                             |
| 'airline_sentiment_confidence' | Confidence with which the given sentiment was determined         |
| 'negativereason'               | Reason for which user posted a negative tweet                    |
| 'negativereason_confidence'    | Confidence with which the negative reason of tweet was predicted |
| 'airline'                      | Airline for which the tweet was posted                           |
| 'airline_sentiment_gold'       |                                                                  |
| 'name'                         | Name of the person who tweeted                                   |
| 'negativereason_gold'          |                                                                  |
| 'retweet_count'                | Number of retweets                                               |
| 'text'                         | Text of the tweet whose sentiment has to be predicted            |
| 'tweet_coord'                  |                                                                  |
| 'tweet_created'                | Time at which the tweet was created                              |
| 'tweet_location'               | Location from where the tweet was posted                         |
| 'user_timezone'                |                                                                  |
|                                |                                                                  |

# %% [markdown]
| Quantitative: | Continuous                                                                   | Discrete |
| ------------- | ---------------------------------------------------------------------------- | -------- |
|               | 'airline_sentiment_confidence', 'negativereason_confidence', 'retweet_count' |    'tweet_created'      |
|               |                                                                              |          |

| Categorical: | Ordinal                                                        | Nominal                    |
| ------------ | -------------------------------------------------------------- | -------------------------- |
|              | 'arline_sentiment', 'negativereason', 'airline_sentiment_gold' | 'airline', 'user_timezone' |
|              |                                                                |                            |

# # %% [markdown] 
# 1. Negative, neutral, positive sentiments of airlines 
# 2. Negative reasons of airlines in question and it's implications 
# 3. Negative Reason Confidence and what this tells us
# 4. Airline sentiment confidence and what this tells us

# %%
df.info()
# %% 
## Type conversions
df['tweet_created'] = df['tweet_created'].astype('datetime64[ns]')
# %%
## Count & proportion of Target variable 
base_color = sb.color_palette()[0]
sb.countplot(data = df, x= 'airline_sentiment', color = base_color,
             order = ['positive', 'neutral', 'negative']);

### add annotations
n_points = df.shape[0]
cat_counts = df['airline_sentiment'].value_counts()
locs, labels = plt.xticks()  # get the current tick locations and labels

### loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    #### get the text property for the label to get the correct count
    count = cat_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/n_points)

    #### print the annotation just below the top of the bar
    plt.text(loc, count-559, pct_string, ha='center', color='w')

# %% [markdown]
# Given that this problem will be turn into a classification problem, 
# the skew to the right toward negative sentiment poses a class impbalance problem. 
# Training a model on an imbalanced dataset will result in a higher accuracy score, with most of the instances as 'negative' sentiment, 
# where the final model will predict one class regardless of the data it's asked to predict. 

# We won't be using accuracy as a metric as a result of it's misleadingness, and instead observe model performance
# using a confusion matrix, and precision, recall, and f1 scores. Additonally, we will look at Cohen's kappa and ROC curves
# to compare classification accuracy. 

# In the event resampling is needed, we will use SMOTE, though non-linear relationships between attributes may not be perserved. 
# Otherwise, we can penalize the class using penalized-SVM and penalized-LDA and notice if the penalties
# bias the model to pay more attention to the positive, and neutral minority classes.  

# %%
## Counting missing data
df.isna().sum()
na_counts = df.isna().sum()
base_color = sb.color_palette()[0]
sb.barplot(na_counts, na_counts.index.values, color=base_color)

# Columns airline_sentiment_gold, negativereason_gold and tweet_coord all have missing values >90%. 

# %% 
## Removing columns: For columns 'tweet_coord', 'negativereason_gold', 'airline_sentiment_gold', >90% of it's data is missing. Unable to determine the meaning or quantitative value of '_gold'.
df.drop(columns=['tweet_coord', 'negativereason_gold', 'airline_sentiment_gold'], axis=1, inplace=True)
# %%
df.loc[df['airline_sentiment'] == 'negative', ['tweet_id','airline_sentiment','text']].sample(n=3)
# %% 
## Exploring messages of tweets by sentiment 
### What 'Neutral' messages look like: 
# %%
df.loc[[12802,10417,4427], ['airline_sentiment', 'airline_sentiment_confidence', 'airline', 'name', 'text', 'user_timezone', 'tweet_created']]
# %%
### What 'Positive' messages look like:
df.loc[[4336, 14325, 998], ['airline_sentiment', 'airline_sentiment_confidence', 'airline', 'name', 'text', 'user_timezone', 'tweet_created']]
#%%
### What 'Negative' messages look like: 
df.loc[[1841, 10040, 5071], ['airline_sentiment', 'airline_sentiment_confidence', 'airline', 'name', 'text', 'user_timezone', 'tweet_created']]
# %%
## Negative reasons 
df['negativereason'].value_counts()
# %%
## Observing reactions from account names with several messages 
df['name'].value_counts()
df[df['name'] == 'itsropes']
## Message count from users over a typical day
df['day'] = df['tweet_created'].apply(lambda x: "%d" % (x.day))
result = df.groupby(['day', 'name']).size()
print("\n Number of tweets by user for each day")
print(result)
# %%
