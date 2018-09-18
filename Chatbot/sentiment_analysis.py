import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re

data = pd.read_csv('sentiment.csv')
data = data[['text', 'sentiment']]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

tok = Tokenizer(num_words=2000, split=' ')
tok.fit_on_texts(data['text'].values)
X = tok.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

model = Sequential()
model.add(Embedding(2000, 128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=50)

model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=2)
model.save('Senti.h5')

score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=32)
print("Score of our model: %.2f" % (score))
print("Accuracy of our model: %.2f" % (acc))
