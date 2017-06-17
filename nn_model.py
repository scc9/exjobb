from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random

class Bandit():
    def __init__(self,no_arms=10,explorer='greedy'):
        self.models=[self.new_model() for i in range(no_arms)]
        self.explorer=explorer
        self.no_arms=no_arms
        self.XX=[[] for i in range(no_arms)]
        self.YY=[[] for i in range(no_arms)]
        self.rng=True

    def new_model(self):
        model = Sequential()
        model.add(Dense(16,input_shape=(10,), activation='relu'))
        model.add(Dense(8,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
        return model

    def recommend(self,features):
        pp=[]
        if self.rng:
            return np.random.randint(self.no_arms)
        elif self.explorer.lower()=='greedy':
            for model in self.models:
                pp.append(model.predict(features.reshape(1,10)))
            return np.argmax(pp)
        else:
            print("NO EXPLORER!!!!")
            return 991.0

    def update_stats(self,arm,features,fb):
        self.XX[arm].append(features)
        self.YY[arm].append(fb)

    def update_model(self):
        self.rng=False  ## NO longer random arm selection
        arm=0
        for model in self.models:
            model.fit(np.array(self.XX[arm],dtype='int'),self.YY[arm],epochs=3,batch_size=16)
            arm+=1
        print("NN UPDATE DONE")
    def reset_stats(self):
        self.models=[self.new_model() for i in range(self.no_arms)]
        self.XX=[[] for i in range(self.no_arms)]
        self.YY=[[] for i in range(self.no_arms)]
        self.rng=True
