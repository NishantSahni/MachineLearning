import gensim
import csv
import nltk
import numpy as np
import pickle

nltk.download('punkt')

l=[]

word_to_vec= gensim.models.KeyedVectors.load_word2vec_format('glove.6B.300d.txt.word2vec', binary=False)

data_set=[]

with open('output.csv','r',encoding='utf-8') as f:
    lines=csv.reader(f)


    for row in lines:
        data_set.append(row)
    f.close()


    ques=[]
    ans=[]

    for row in data_set:
        ques.append(row[0])
        ans.append(row[1])


    tok_ques=[]
    tok_ans=[]

    for i in range(len(ques)):
        tok_ques.append(nltk.word_tokenize(ques[i]))
        tok_ans.append(nltk.word_tokenize(ans[i]))

    vec_ques=[]
    vec_ans=[]

    for sublist in tok_ques:
        tempword = []
        for word in sublist:
            if(word in word_to_vec.vocab):
                tempword.append(word_to_vec[word])
        vec_ques.append(tempword)


    for sublist in tok_ans:
        tempword = []
        for word in sublist:
            if (word in word_to_vec.vocab):
                tempword.append(word_to_vec[word])
        vec_ans.append(tempword)

    pad_vec = np.ones((300,), dtype=np.float32)

    pad_vec=np.array([pad_vec])

    vec_ans_new=[]
    vec_ques_new=[]

    for i in range(len(vec_ans)):
        if len(vec_ans[i])!=0:
            vec_ans_new.append(vec_ans[i])
            vec_ques_new.append(vec_ques[i])


    for i in range(len(vec_ques_new)):
        if (len(vec_ques_new[i]) > 15):
            for j in range(len(vec_ques_new[i]) - 15):
                vec_ques_new[i].pop()

    for i in range(len(vec_ques_new)):
        if (len(vec_ques_new[i]) < 15):
            vec_len=len(vec_ques_new[i])
            for j in range(15 - vec_len):
                vec_ques_new[i]=np.append(vec_ques_new[i],pad_vec,axis=0)

    for i in range(len(vec_ans_new)):
        if(len(vec_ans_new[i])>15):
            for j in range(len(vec_ans_new[i]) - 15):
                vec_ans_new[i].pop()

        if (len(vec_ans_new[i]) < 15):
            vec_len=len(vec_ans_new[i])
            for j in range(15 - vec_len):
                vec_ans_new[i]=np.append(vec_ans_new[i],pad_vec,axis=0)

    with open('final.pickle','wb') as f:
        pickle.dump([vec_ques_new,vec_ans_new],f)
