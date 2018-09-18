from scipy import spatial
import numpy as np
import gensim
import nltk
import sys
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

model = load_model('convo3.h5')
senti = load_model('Senti.h5')

word2vec = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.300d.txt.word2vec',binary=False)
while(True):
    input_text = input('Enter your message: ')
    tokenizer = Tokenizer(num_words=15, split=' ')
    tokenizer.fit_on_texts(input_text)
    tokens = tokenizer.texts_to_sequences(input_text)
    tokens = pad_sequences(tokens, maxlen=40, dtype='float', padding='post', truncating='post', value=0.5)
    sentiment = senti.predict(tokens, batch_size=1, verbose=2)[0]

    if (np.argmax(sentiment) == 0):
        print("Negative")
    elif (np.argmax(sentiment) == 1):
        print("Positive")

    pad_vec = np.ones((300,), dtype=np.float32)
    msg = nltk.word_tokenize(input_text.lower())
    input_vector=[]
    tempword = []
    for word in msg:
        if (word in word2vec.vocab):
            input_vector.append([word2vec[word]])

    input_vector[14:] = []
    if len(input_vector) < 15:
        for i in range(15 - len(input_vector)):
            input_vector.append([pad_vec])
    input_vector = np.array(input_vector)
    input_vector=np.reshape(input_vector,(1,15,300))

    pred = model.predict(input_vector)
    response = []
    #print(pred)

    for k in range(15):
        response.append(word2vec.wv.most_similar([pred[0][k]])[0][0])
    
    final_response = ' '.join(response)
    print(final_response)
