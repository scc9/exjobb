from __future__ import division
import numpy as np


class LINUCB():

    def __init__(self,alpha,no_features,no_arms):
        self.alpha=alpha
        self.d=no_features
        self.observed_articles=[]
        self.AA=[]
        self.bb=[]
        self.AA_inv=[]
        self.ww=[]
        self.R=0
        self.T=0
        self.R_tot=0
        self.T_tot=0
        for i in range(no_arms):
            self.add_arm(i)


    def init_A(self):
        A=np.eye(self.d)
        return A

    def init_b(self):
        b=np.zeros([self.d,1])
        print "WTF?"
        return b

    def param_update(self,arm):
        self.AA_inv[arm]=np.linalg.inv(self.AA[arm])
        self.ww[arm]=np.matmul(np.linalg.inv(self.AA[arm]),self.bb[arm])


    def update_A(self,arm,x):
        self.AA[arm]=np.add(self.AA[arm],np.matmul(x,np.transpose(x)))
        #print np.matmul(x,np.transpose(x))


    def update_b(self,arm,x,r):
        self.bb[arm]=np.add(self.bb[arm],np.multiply(r,x.reshape(10,1)))


    def predict(self,arm,x):
        p=np.add(np.matmul(np.transpose(self.ww[arm]),x),np.multiply(self.alpha,np.sqrt(np.matmul(np.matmul(np.transpose(x),self.AA_inv[arm]),x))))
        #print np.matmul(np.transpose(w),x)
        p1=np.matmul(np.transpose(self.ww[arm]),x)

        #print self.ww[arm].shape
        #print self.ww[0].shape
        return p

    def recommend(self,user_features):
        user_features=user_features.reshape(self.d,1)
        no_arms=len(self.observed_articles)
        pp=np.zeros(no_arms)

        for i in range(no_arms):
            pp[i]=self.predict(i,user_features)
        top=np.argmax(pp)
        return top

    def update_stats(self,arm,user_features,click):
        user_features=user_features.reshape(self.d,1)
        self.update_A(arm,user_features)
        self.update_b(arm,user_features,click)
        self.update_w_A_inv(arm)

        self.R_tot+=click
        self.T_tot+=1
        self.R+=click
        self.T+=1
        if self.T==1000:

            print "Last 1000 linUCB CTR: " + str(self.R/self.T)
            print "linUCB CTR after " + str(self.T_tot) +" observations: " + str(self.R_tot/self.T_tot)
            self.R=0
            self.T=0

    def update_w_A_inv(self,arm):
        self.AA_inv[arm]=np.linalg.inv(self.AA[arm])
        self.ww[arm]=np.matmul(self.AA_inv[arm],self.bb[arm])
    def add_arm(self,article_id):
        self.observed_articles.append(article_id)
        self.AA.append(np.eye(self.d))
        self.AA_inv.append(np.eye(self.d))
        self.bb.append(np.zeros([self.d,1]))
        self.ww.append(np.zeros([self.d,1]))

    def reset_stats(self):
        for i in range(len(self.observed_articles)):
            self.AA[i]=np.eye(self.d)
            self.AA_inv[i]=np.eye(self.d)
            self.bb[i]=np.zeros([self.d,1])
            self.ww[i]=np.zeros([self.d,1])
        self.R=0
        self.T=0
        self.R_tot=0
        self.T_tot=0
