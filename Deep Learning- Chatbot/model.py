from keras import Sequential
import numpy as np
import pickle
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split

with open('final.pickle','rb') as f:
    vec_ques,vec_ans=pickle.load(f)

x_train,x_test,y_train,y_test=train_test_split(vec_ques,vec_ans,test_size=0.01)

x_train_new=[]
y_train_new=[]

#print(len(x_train))
#print(len(y_train))

for i in range(len(x_train)):
    x_train_new.append(x_train[i])
    y_train_new.append(y_train[i])

x_train_new=np.array(x_train_new)
y_train_new=np.array(y_train_new)

x_train_new=np.reshape(x_train_new,(len(x_train_new),15,300))
y_train_new=np.reshape(y_train_new,(len(y_train_new),15,300))


print(x_train_new.shape)
print(y_train_new.shape)

model=Sequential()
model.add(LSTM(300, input_shape=(15,300), activation='softmax', kernel_initializer='glorot_normal', bias_initializer='glorot_normal', recurrent_initializer='glorot_normal', return_sequences=True))
#model.add(Dropout=0.2)
model.add(LSTM(300, return_sequences=True, activation="softmax", kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal"))
#model.add(Dropout=0.2)
model.add(LSTM(300, return_sequences=True, activation="softmax", kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal"))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train_new,y_train_new,nb_epoch=50,validation_data=(np.array(x_test),np.array(y_test)))
model.save('final1.h5')
model.fit(x_train_new,y_train_new,nb_epoch=50,validation_data=(np.array(x_test),np.array(y_test)))
model.save('final2.h5')
model.fit(x_train_new,y_train_new,nb_epoch=50,validation_data=(np.array(x_test),np.array(y_test)))
model.save('final3.h5')
#model.fit(x_train_new,y_train_new,nb_epoch=50,validation_data=(np.array(x_test),np.array(y_test)))
#model.save('LSTM1000.h5')
#model.fit(x_train_new,y_train_new,nb_epoch=50,validation_data=(np.array(x_test),np.array(y_test)))
#model.save('LSTM1250.h5')
