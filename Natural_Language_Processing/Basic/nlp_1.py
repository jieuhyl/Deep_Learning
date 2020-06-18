# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:55:27 2020

@author: Jie.Hu
"""

#import nltk
#nltk.download()
#import gensim
''' Tokenize '''
from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

# Define input text
input_text = "Do you know how tokenization works? It's actually quite interesting! Let's analyze a couple of sentences and figure it out." 

# Sentence tokenizer
print("\nSentence tokenizer:")
print(sent_tokenize(input_text))

# Word tokenizer
print("\nWord tokenizer:")
print(word_tokenize(input_text))

# WordPunct tokenizer
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))




''' Stemming '''
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 
        'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

# Create various stemmer objects
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english')

# Create a list of stemmer names for display
stemmer_names = ['PORTER', 'LANCASTER', 'SNOWBALL']
formatted_text = '{:>16}' * (len(stemmer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *stemmer_names), 
        '\n', '='*68)

# Stem each word and display the output
for word in input_words:
    output = [word, porter.stem(word), 
            lancaster.stem(word), snowball.stem(word)]
    print(formatted_text.format(*output))
    
    
    
    
''' Lemmatization '''
from nltk.stem import WordNetLemmatizer

input_words = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize', 
        'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']

# Create lemmatizer object 
lemmatizer = WordNetLemmatizer()

# Create a list of lemmatizer names for display
lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *lemmatizer_names), 
        '\n', '='*75)

# Lemmatize each word and display the output
for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos='n'),
           lemmatizer.lemmatize(word, pos='v')]
    print(formatted_text.format(*output))    
    
    
    


''' Chunks '''
import numpy as np
from nltk.corpus import brown

# Split the input text into chunks, where
# each chunk contains N words
def chunker(input_data, N):
    input_words = input_data.split(' ')
    output = []

    cur_chunk = []
    count = 0
    for word in input_words:
        cur_chunk.append(word)
        count += 1
        if count == N:
            output.append(' '.join(cur_chunk))
            count, cur_chunk = 0, []

    output.append(' '.join(cur_chunk))

    return output 

if __name__=='__main__':
    # Read the first 12000 words from the Brown corpus
    input_data = ' '.join(brown.words()[:12000])

    # Define the number of words in each chunk 
    chunk_size = 700

    chunks = chunker(input_data, chunk_size)
    print('\nNumber of text chunks =', len(chunks), '\n')
    for i, chunk in enumerate(chunks):
        print('Chunk', i+1, '==>', chunk[:50])
        
        
        


''' Bag of Words '''
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
#from text_chunker import chunker 

# Read the data from the Brown corpus
input_data = ' '.join(brown.words()[:5400])

# Number of words in each chunk 
chunk_size = 800

text_chunks = chunker(input_data, chunk_size)

# Convert to dict items
chunks = []
for count, chunk in enumerate(text_chunks):
    d = {'index': count, 'text': chunk}
    chunks.append(d)

# Extract the document term matrix
count_vectorizer = CountVectorizer(min_df=7, max_df=20)
document_term_matrix = count_vectorizer.fit_transform([chunk['text'] for chunk in chunks])

# Extract the vocabulary and display it
vocabulary = np.array(count_vectorizer.get_feature_names())
print("\nVocabulary:\n", vocabulary)

# Generate names for chunks
chunk_names = []
for i in range(len(text_chunks)):
    chunk_names.append('Chunk-' + str(i+1))

# Print the document term matrix
print("\nDocument term matrix:")
formatted_text = '{:>12}' * (len(chunk_names) + 1)
print('\n', formatted_text.format('Word', *chunk_names), '\n')
for word, item in zip(vocabulary, document_term_matrix.T):
    # 'item' is a 'csr_matrix' data structure
    output = [word] + [str(freq) for freq in item.data]
    print(formatted_text.format(*output))

        
        