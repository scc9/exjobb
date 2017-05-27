from __future__ import division
import numpy as np
from random import shuffle
from random import gauss
from random import normalvariate
from scipy import optimize
from sklearn.linear_model import SGDClassifier
import time
import random



class Model():

    def __init__(self,lamb,dim,no_arms,explorer,epsilon):
        self.epsilon=epsilon
        self.explorer=explorer
        self.no_arms=no_arms
        self.rng=True
        self.dim=dim
        self.lamb=lamb
        self.observed_articles=[]
        self.R=0
        self.T=0
        self.R_tot=0
        self.T_tot=0
        #-----Thompson params----
        self.CLASSIFIER=[] #weights
        self.QQ=[]  #precision for tmp sampling
        self.WW=[] #sampled weights (only re sampled if they are consumed by pseudo stream)
        self.XX=[]
        self.YY=[]
        for i in range(no_arms):
            self.add_arm(i)



    def add_arm(self,article_id):
        self.observed_articles.append(article_id)
        self.CLASSIFIER.append(SGDClassifier(loss="log",penalty="l2",n_iter=1000,fit_intercept=False,shuffle=True,warm_start=True)) # constante feature generates intercept
        self.QQ.append(self.lamb*np.ones([1,self.dim]))
        self.WW.append([])
        self.XX.append([]) #store data for batch updates
        self.YY.append([])
    def reset_stats(self):
        for i in range(self.no_arms):
            self.QQ[i]=self.lamb*np.ones([1,self.dim])
            self.WW[i]=[]
            self.XX[i]=[] #store data for batch updates
            self.YY[i]=[]
    def shuffle_batch(self,arm,batch_size):
        no_items=len(self.YY[arm])
        if no_items<batch_size:
            print("Warning: Requested batch of size " + str(batch_size) + " from pool of only " + str(no_items) +" samples!")
            batch_size=no_items
            print("batch_size set to no_items")

        X_shuf=np.empty([batch_size,self.dim])
        Y_shuf = np.empty([batch_size,])
        index_shuf = list(range(no_items))
        shuffle(index_shuf)
        for i in range(batch_size):
            Y_shuf[i]=self.YY[arm][index_shuf[i]]
            X_shuf[i]=self.XX[arm][index_shuf[i]]
        return X_shuf, Y_shuf

    def update_arm(self,arm,no_iter=50):

            no_items=len(self.YY[arm])

            #t=time.time()
            if sum(self.YY[arm])>0:
                self.CLASSIFIER[arm].fit(X=self.XX[arm],y=self.YY[arm])
            else:
                for i in range(no_iter):

                    X_shuf,Y_shuf=self.shuffle_batch(arm,no_items)
                    self.CLASSIFIER[arm].partial_fit(X=X_shuf,y=Y_shuf,classes=np.array([0,1]))
            #elapsed_t=time.time()-t
            #print elapsed_t

            ##Q comp with nice matrix mult
            preds=self.CLASSIFIER[arm].predict_proba(self.XX[arm])

            p=np.multiply(preds[:,0],preds[:,0]).reshape(-1,no_items)
            q=np.matmul(p,self.XX[arm])

            self.QQ[arm]+=q*0.1



    def init_collect(self,rtcl,x,y):
        if (rtcl not in self.observed_articles):
            self.add_arm(rtcl)

        top=self.observed_articles.index(rtcl)
        self.XX[top].append(x)
        self.YY[top].append(y)

    def init_train(self):
        for arm in range(len(self.observed_articles)):

            self.CLASSIFIER[arm].fit(X=self.XX[arm],y=self.YY[arm])

    def predict(self,arm,x):

        #pred=self.CLASSIFIER[arm].predict_proba(x.reshape(1,-1))
        #print pred
        #print pred[0][1]
        #p=pred[0][1]

        #if not self.WW[arm]:
        if self.rng:
            return random.random()

        else:
            self.sample_params(arm)
            #print self.CLASSIFIER[arm].predict_proba(x)[0][1]

            '''
            EXPLORATION ISSUE TO BE RESOLVED
            '''
            #return self.CLASSIFIER[arm].predict_proba(x.reshape(1,-1))[0][1]
            return np.divide(1.0,np.add(1,np.exp(-np.matmul(self.WW[arm],x))))

    def sample_params(self,arm):

        mean=self.CLASSIFIER[arm].coef_
        var=np.divide(1,self.QQ[arm])
        w=[np.random.normal(mu,np.sqrt(var)) for mu, var in zip(mean,var)]

        self.WW[arm]=w

    def update_stats(self,arm,user_features,feed_back):
        '''
        Appends observed features and feedback.
        Does not train the model with respect to these obervations!
        '''
        self.XX[arm].append(user_features)
        self.YY[arm].append(feed_back)
        self.R_tot+= feed_back==1
        self.T_tot+=1
        self.R+=feed_back==1
        self.T+=1

    def update_model(self):
        '''
        Trains model on all observations.
        '''
        print("TRAINING LOGISTIC REGRESSION MODEL WITH SGD")
        for arm in self.observed_articles:

            self.update_arm(arm)
            self.sample_params(arm)


    def performance(self,arm,x,y):
        #TODO
        y_=self.CLASSIFIER[arm].predict_proba(x)


    def recommend(self,user_features):


        if self.T==1000:
            print("Last 1000 Thompson CTR: " + str(self.R/self.T))
            print("Thompson CTR after " + str(self.T_tot) +" observations: " + str(self.R_tot/self.T_tot))
            #print "has real features ratio: " + str(self.more/(self.ett+self.more))
            #print pp
            #print [self.CLASSIFIER[arm].predict_proba(user_features.reshape(1,-1))[0,1] for arm in range(len(self.observed_articles)) ]
            print([len(self.YY[arm]) for arm in range(len(self.observed_articles)) ])
            print([sum(self.YY[arm]) for arm in range(len(self.observed_articles)) ])
            #print(pp)
            self.R=0
            self.T=0

        pp=np.zeros(len(self.observed_articles))

        for i in self.observed_articles:
            #if len(self.YY[i])>=1:
            #    self.sample_params(i)

            pp[i]=self.predict(i,user_features)
    #    print pp
        if self.explorer=='greedy':
            return np.argmax(pp)

        elif self.explorer=='epsilon':
            if random.random()<=self.epsilon:
                return random.choice(self.observed_articles)
            else:
                return np.argmax(pp)

        elif self.explorer=='boltzman':
            p1=np.exp(pp/self.epsilon)
            p_sum=np.sum(p1)
            p_dist=p1/p_sum
            return np.argmax(np.random.multinomial(1,p_dist,size=1))
        else:
            print("WARNING: NO EXPLORER GIVEN")

            return 0








'''
tomp_test=Model(1,10,1)
x=[1,0,1,0,1,0,1,1,1,1]
x=np.array(x).reshape([10])
x2=[1,1,0,1,0,0,0,0,0,0]
x2=np.array(x2).reshape([10])
for kk in range(100):

    arm= tomp_test.recommend(x2)
    for i in range(10):
        tomp_test.update_stats(arm,x2,0)
        tomp_test.update_stats(arm,x,1)
    tomp_test.update_stats(arm,x2,1)
    tomp_test.update_stats(arm,x,0)
    #print np.divide(1,1+np.exp(-np.matmul(np.transpose(tomp_test.WW[0]),x)))
    #print np.divide(1,1+np.exp(-np.matmul(np.transpose(tomp_test.WW[0]),x2)))


w=tomp_test.CLASSIFIER[arm].coef_

print tomp_test.predict(arm,x2)
print tomp_test.predict(arm,x)
print w
print "pred!"
print np.divide(1,1+np.exp(-np.matmul(w,x2.reshape([10,1]))))
print tomp_test.CLASSIFIER[arm].predict_proba(x2)
x=tomp_test.XX[arm]
preds= tomp_test.CLASSIFIER[arm].predict_proba(x)
print preds.shape
print preds[:,1]
#print tomp_test.CLASSIFIER[arm].decision_function(x2)

print tomp_test.CLASSIFIER[arm].get_params(deep=True)
print "Done!"
print "NITER IS HIGH"
#print tomp_test.QQ[0]
'''
