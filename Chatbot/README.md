# ANA-Interactive-SmartBot
The aim of this project is to develop a bot which will successfully answer questions asked to ‘it’ and engage in light conversation relating to that topic. Using a corpus with question and answers, we developed a neural network using Keras as our model for training our bot. This will allow it to successfully answer questions and engage in conversation with the user.

All the python code is written in Python 3.6 and will require the following dependencies to run: NLTK, Gensim, Numpy, Pickle, TensorFlow, 
Keras and ScikitLearn.

The files can be run by typing "python3 file_name.py" into the terminal once in the same directory as the files.

Please find as follows a description of each file.

1. data_prep.py: This file is responsible for combining the SQuAD and Carnegie Mellon datasets into one single dataset before 
preprocessing.

2. preprocessing.py: This file is responsible for converting the data to lower case, tokenizing, padding the data and obtaining a vector
 representation of each word before the model is trained. This creates a pickle dump of the vectorized data which is used by model.py.


3. model.py: This file loads the preprocessed and vectorized pickle dump file and creates a Recurrent Neural Network model using LSTMs. 
This model is then stored as a .h5 file and is used by front_end.py to predict responses.


4. sentiment_analysis.py: This file loads the movie review dataset, preprocesses the data, vectorizes it and creates a Recurrent 
Neural Network using LSTMs for training a model for sentiment analysis. This model is then saved as a .h5 file and is used by 
front_end.py to determine the sentiment of the user's message.

5. front_end.py: This file is the front end interface that the user interacts with. It loads both, the trained chatbot model and 
the trained sentiment analysis model and predicts the response and sentiment of a user's message.

6. glove.6B.300d.txt.word2vec: This is the word2vec file that we create based on the GloVe vector file and has 300 vectors for 
each word.

7. output.csv: This file contains the combined SQuaD and Carnegie Mellon datasets.

8. sentiment.csv: This file contains the movie review data for sentiment analysis.

9. conversation.json: This file contains the conversation data to train the chatbot with.
