import csv
import json as js
import pandas as pd
import numpy as np

class chatbot():

    def __init__(self):
        self.training_set = []
        self.normalized_data=[]

    def input(self):

        with open('training.csv','r',encoding="utf8") as f:
            lines=csv.reader(f)
            count=0
            for row in lines:
                count=count+1
        f.close()


        with open('training.csv','r',encoding="utf8") as f:
            lines=csv.reader(f)
            prev=""
            c=0
            for row in lines:
                if(row==prev):
                    prev=row
                    continue
                self.training_set.append(row)
                c=c+1
                prev=row
            print(self.training_set[1])

        self.normalized_data=[[] for j in range(c)]

        f.close()
        #lowercase
        i=0
        for row in self.training_set:
            for elements in row[0:3]:
                self.normalized_data[i].append(elements.lower())
            i=i+1

        #null
        for row in self.normalized_data:
            for i,elements in enumerate(row[0:3]):
                if(elements==''):
                    row[i]='null'

        for row in self.normalized_data:
            print(row)

        print(len(self.normalized_data))
        print(str(count) + " " +str(c))




if __name__ == '__main__':
    cb =chatbot()
    cb.input()

jsonData = js.load(open("train-v1.1.json"))

data_list = []

for i in range(len(jsonData['data'])):
    for j in range(len(jsonData['data'][i]['paragraphs'])):
        for k in range(len(jsonData['data'][i]['paragraphs'][j]['qas'])):
            data_list.append([jsonData['data'][i]['title'].lower(),
                              jsonData['data'][i]['paragraphs'][j]['qas'][k]['question'].lower(),
                              jsonData['data'][i]['paragraphs'][j]['qas'][k]['answers'][0]['text'].lower()])

df = pd.DataFrame(np.array(data_list).reshape(len(data_list), 3), columns=['Topic', 'Question', 'Answer'])
#print(df.to_string())
df.to_csv('squad.csv')
