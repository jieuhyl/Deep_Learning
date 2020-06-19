# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:57:45 2020

@author: Jie.Hu
"""


import spacy
#import spacy.cli
#spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')
nlp.vocab.vectors.shape

# word
nlp(u'tiger').vector
nlp(u'lion').vector
# sentence
nlp(u'tiger lion').vector
doc = nlp(u'The quick brown fox jumped over the lazy dogs.')
doc.vector

# Create a three-token Doc object:
tokens = nlp(u'lion tiger cat pet')


# Similar vectors
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
        
        
nlp(u'tiger').similarity(nlp(u'lion'))


# Vector norms
tokens = nlp(u'dog cat nargle')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    
    
    
# Vector arithmetic    
from scipy import spatial

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

#king = nlp(u'king').vector
#man = nlp(u'man').vector
#woman = nlp(u'woman').vector

# Now we find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])    


import pandas as pd
df = pd.read_csv('labels.csv')

target = nlp(df['Text'][0])
 
doc1 = nlp(df['Text'][1])
doc2 = nlp(df['Text'][2])
 
print(target.similarity(doc1))  
print(target.similarity(doc2)) 

target = nlp(df['Text'][0])
for i in range(1, len(df)):
    doc = nlp(df['Text'][i])
    print(target.similarity(doc))
    
vec_space = []
for i in range(len(df)):
    vec_space.append(nlp(df['Text'][i]).vector)    

df['Vec'] = vec_space
df.head()

for i in range(len(df['Vec'][0])):
    df['vec{}'.format(str(i+1))]=[float(k[i]) for k in df['Vec']]

df.to_csv('label_v1.csv', index=False)


df = pd.read_csv('label_v1.csv')

from sklearn.metrics.pairwise import cosine_similarity
mtx = cosine_similarity(df.iloc[:, 3:])


import seaborn as sns
sns.heatmap(mtx, 
            xticklabels=df['Label'].tolist(), 
            yticklabels=df['Label'].tolist(), 
            annot=False)


from sklearn.manifold import TSNE
X = df.iloc[:, 3:].values
tsne = TSNE(n_components=2, 
            #perplexity = 10, 
            #early_exaggeration = 20, 
            #learning_rate = 200, 
            random_state = 1337)
result = tsne.fit_transform(X) 

#df_trans = pd.DataFrame(result)
df['D1'] = result[:, 0]
df['D2'] = result[:, 1]
sns.scatterplot(x="D1", y="D2", data=df)


import seaborn as sns
import matplotlib.pyplot as plt

def scatter_text(x, y, text_column, data, title, xlabel, ylabel):
    """Scatter plot with country codes on the x y coordinates
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    # Create the scatter plot
    p1 = sns.scatterplot(x, y, data=data, size = 8, legend=False)
    # Add text besides each point
    for line in range(0,data.shape[0]):
         p1.text(data[x][line]+0.01, data[y][line], 
                 data[text_column][line], horizontalalignment='left', 
                 size='medium', color='black', weight='semibold')
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1

plt.figure(figsize=(20,10))
scatter_text('D1', 'D2', 'Label',
             data = df, 
             title = 'TITLE', 
             xlabel = 'D1',
             ylabel = 'D2')
