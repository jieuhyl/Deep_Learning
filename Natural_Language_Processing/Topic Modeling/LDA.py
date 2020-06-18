# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 01:54:12 2020

@author: Jie.Hu
"""


import pandas as pd
npr = pd.read_csv('npr.csv')
# Randomly sample 10% of your dataframe
npr = npr.sample(frac=0.1)
# Randomly sample 100 elements from your dataframe
#npr = npr.sample(n=100)

npr.head()

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = count.fit_transform(npr['Article'])


from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7, random_state=42)
LDA.fit(dtm)

len(count.get_feature_names())
len(LDA.components_)

single_topic = LDA.components_[0]
# Returns the indices that would sort this array.
single_topic.argsort()
single_topic.argsort()[0]
single_topic.argsort()[-1]
# Top 10 words for this topic:
single_topic.argsort()[-10:]

top_word_indices = single_topic.argsort()[-10:]
for index in top_word_indices:
    print(count.get_feature_names()[index])
    
    
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([count.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


topic_results = LDA.transform(dtm)
topic_results.shape
topic_results[0]
topic_results[0].round(2)
topic_results[0].argmax()


npr.head()
topic_results.argmax(axis=1)
npr['Topic'] = topic_results.argmax(axis=1)
npr.head(10)
    