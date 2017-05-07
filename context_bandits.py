from __future__ import division
import numpy as np
import random
import scipy as sp
import matplotlib.pyplot as plt
import linUCB_model as ucb

'''
Multi armed bandit test bench with support for multi contexts

'''
#np.random.seed(2)
#random.seed(2)


class Bandit():

    def __init__(self,no_arms=5,no_contexts=3):
        #List of list of lists
        self.no_contexts=no_contexts
        self.bern_p=[np.random.choice([0.01, 0.05, 0.09], size=(no_contexts,), p=[3./5, 1./5, 1./5 ]) for i in range(no_arms)]
        self.no_arms=no_arms
        self.contexts=[np.append(1,np.random.choice([0.9, 0.1], size=(9,), p=[1./2, 1./2 ])) for j in range(no_contexts)]
        temp=[]
        p1=0
        p2=0
        p3=0
        for probs in self.bern_p:
            p1=max(p1,probs[0])
            p2=max(p2,probs[1])
            p3=max(p3,probs[2])

        self.omni=np.mean([p1,p2,p3])
    def context_gen(self):
        ctxt=np.random.choice(range(self.no_contexts))
        noisy_features=self.contexts[ctxt]>= np.random.rand(10) # NOISE np.ones(10)*0.5
        return noisy_features,ctxt



    def feed_back(self,arm,ctxt):
        return random.random()<=self.bern_p[arm][ctxt]

class HyperTS():
    def __init__(self,arm_list,alpha=0.5,beta=0.5):
        self.no_arms=len(arm_list)
        self.beta=beta
        self.curr_arm=0
        self.alpha=alpha
        self.no_success=np.ones(self.no_arms)*alpha
        self.no_fails=np.ones(self.no_arms)*beta
        self.arm_list=arm_list

    def recommend(self,features):
        p=[np.random.beta(self.no_success[arm],self.no_fails[arm]) for arm in range(self.no_arms)]
        arm=np.argmax(p)
        self.curr_arm=arm
        return self.arm_list[arm].recommend(features)

    def update_stats(self,arm,features,feedback):
        self.no_fails[self.curr_arm]=self.no_fails[self.curr_arm]*0.9+ (1-feedback)*10
        self.no_success[self.curr_arm]=self.no_success[self.curr_arm]*0.9 + feedback*10
        for alg in self.arm_list:
            alg.update_stats(arm,features,feedback)

    def reset_stats(self):
        self.no_success=np.ones(self.no_arms)*self.alpha
        self.no_fails=np.ones(self.no_arms)*self.beta
        for arm in self.arm_list:
            arm.reset_stats()


class Thompson():
    def __init__(self,no_arms=5,alpha=0.5,beta=0.5):
        self.no_arms=no_arms
        self.beta=beta
        self.alpha=alpha
        self.no_success=np.ones(no_arms)*alpha
        self.no_fails=np.ones(no_arms)*beta
        self.arm_list=range(no_arms)

    def recommend(self,features):
        p=[np.random.beta(self.no_success[arm],self.no_fails[arm]) for arm in self.arm_list]
        return np.argmax(p)

    def update_stats(self,arm,features,feedback):
        self.no_fails[arm]+= (1-feedback)
        self.no_success[arm]+=feedback

    def reset_stats(self):
        self.no_success=np.ones(self.no_arms)*self.alpha
        self.no_fails=np.ones(self.no_arms)*self.beta

class E_greedy():
    def __init__(self,no_arms=5,epsilon=0.1):
        self.no_arms=no_arms
        self.epsilon=epsilon
        self.mean=np.random.rand(no_arms)
        self.no_plays=np.zeros(no_arms)
        self.no_success=np.zeros(no_arms)
        self.arm_list=range(no_arms)

    def recommend(self,features):
        if random.random()<=self.epsilon:
            return random.choice(self.arm_list)
        else:
            return np.argmax(self.mean)

    def update_stats(self,arm,features,feedback):
        self.no_plays[arm]+=1
        self.no_success[arm]+=feedback
        self.mean[arm]=self.no_success[arm]/self.no_plays[arm]

    def reset_stats(self):
        self.mean=np.random.rand(self.no_arms)
        self.no_plays=np.zeros(self.no_arms)
        self.no_success=np.zeros(self.no_arms)

class HyperUCB1():
    def __init__(self,arm_list):
        self.no_arms=len(arm_list)
        self.total_plays=0
        self.mean=np.zeros(self.no_arms)
        self.p=np.random.rand(self.no_arms)*100
        self.no_plays=np.zeros(self.no_arms)
        self.no_success=np.zeros(self.no_arms)
        self.arm_list=arm_list

    def ucb(self,arm):

        return self.mean[arm] + np.sqrt(np.log(self.total_plays)/self.no_plays[arm])

    def recommend(self):
        #p=[self.ucb(arm) for arm in self.arm_list]
        arm=np.argmax(self.p)
        self.curr_arm=arm
        return self.arm_list[arm].recommend()
    def update_stats(self,arm,feedback):
        self.total_plays+=1
        self.no_plays[self.curr_arm]+=1
        self.no_success[self.curr_arm]+=feedback
        self.mean[self.curr_arm]=self.no_success[self.curr_arm]/self.no_plays[self.curr_arm]
        self.p[self.curr_arm]=self.ucb(self.curr_arm)
        for alg in self.arm_list:
            alg.update_stats(arm,feedback)

    def reset_stats(self):
        self.total_plays=0
        self.mean=np.zeros(self.no_arms)
        self.no_plays=np.zeros(self.no_arms)
        self.no_success=np.zeros(self.no_arms)
        self.p=np.random.rand(self.no_arms)*100
        for arm in self.arm_list:
            arm.reset_stats()

class UCB1():
    def __init__(self,no_arms=5):
        self.no_arms=no_arms
        self.total_plays=0
        self.mean=np.zeros(no_arms)
        self.p=np.random.rand(self.no_arms)*100
        self.no_plays=np.zeros(no_arms)
        self.no_success=np.zeros(no_arms)
        self.arm_list=range(no_arms)

    def ucb(self,arm):

        return self.mean[arm] + np.sqrt(np.log(self.total_plays)/self.no_plays[arm])

    def recommend(self,features):
        #p=[self.ucb(arm) for arm in self.arm_list]
        return np.argmax(self.p)

    def update_stats(self,arm,features,feedback):
        self.total_plays+=1
        self.no_plays[arm]+=1
        self.no_success[arm]+=feedback
        self.mean[arm]=self.no_success[arm]/self.no_plays[arm]
        self.p[arm]=self.ucb(arm)

    def reset_stats(self):
        self.total_plays=0
        self.mean=np.zeros(self.no_arms)
        self.no_plays=np.zeros(self.no_arms)
        self.no_success=np.zeros(self.no_arms)
        self.p=np.random.rand(self.no_arms)*100



no_events=10000
no_runs=100
no_arms=10
no_features=10
band=Bandit(no_arms=no_arms)

linUCB_model0=ucb.LINUCB(alpha=0,no_features=no_features,no_arms=no_arms)
linUCB_model01=ucb.LINUCB(alpha=0.1,no_features=no_features,no_arms=no_arms)
linUCB_model1=ucb.LINUCB(alpha=1,no_features=no_features,no_arms=no_arms)
linUCB_model2=ucb.LINUCB(alpha=2,no_features=no_features,no_arms=no_arms)






thomp=Thompson(no_arms=10,alpha=0.5,beta=0.5)
hyperTS=HyperTS([thomp,linUCB_model01,linUCB_model1,linUCB_model2])
#hyperUCB1=HyperUCB1([Thompson(no_arms=10,alpha=0.5,beta=0.5),UCB1(no_arms=10),E_greedy(no_arms=10,epsilon=0.1)])

ucb1=UCB1(no_arms=10)
e_greedy1=E_greedy(no_arms=10,epsilon=0.1)
#avg_ctr=np.zeros(1000)
models=[thomp,linUCB_model01,linUCB_model1,linUCB_model2,hyperTS]
colors=['gray','black','blue','red','green','pink']
linestyles = ['-', '-', '-', '-','--']
labels=['Thompson','linUCB 0.1','linUCB 1', 'linUCB 2','HyperTS']

cind=0
stats_mtrx=np.empty([no_runs,no_events])


for model in models:
    print "New Model"
    print cind
    #best_choice=np.zeros(no_events)
    #spot_ctr=np.zeros(no_events)
    for j in range(no_runs):
        print j
        CTR=[]
        t=0
        r=0

        for i in range(no_events):
            features,ctx=band.context_gen()
            arm=model.recommend(features)
            fb=band.feed_back(arm,ctx)
            model.update_stats(arm,features,fb)

            #arm2=e_greedy.recommend()
            #fb2=band.feed_back(arm2)
            #e_greedy2.update_stats(arm2,fb2)

            #arm=thomp.recommend()
            #fb=band.feed_back(arm)
            #thomp.update_stats(arm,fb)

            #arm=ucb1.recommend()
            #fb=band.feed_back(arm)
            #ucb1.update_stats(arm,fb)
            #best_choice[i]+=arm==9
            #spot_ctr[i]+= (arm+1)/100
            t+=1
            r+=fb
            stats_mtrx[j,i]=r/t


        model.reset_stats()
        #e_greedy.reset_stats()
        #ucb1.reset_stats()
        #thomp.reset_stats()
        #print r/t



    mean=np.empty(no_events)
    std=np.empty(no_events)
    for k in range(no_events):
        mean[k]=np.mean(stats_mtrx[:,k])
        #std[k]=np.std(stats_mtrx[:,k])/2  #half std
    x=range(no_events)
    #up=np.add(mean,std/2)
    mean=mean[0::100]
    mean[0]=0
    x=x[0::100]
    #down=np.subtract(mean,std/2)
    f1 = plt.figure(1)
    plt.plot(x,mean,linewidth=3.0,color=colors[cind],linestyle=linestyles[cind],label=labels[cind])
    #plt.fill_between(x,up,down,facecolor=colors[cind],alpha=0.3)

    cind+=1
plt.plot(x,np.ones(len(x))*band.omni,linewidth=2.0,color="gray",linestyle=':',label="upper limit")
plt.xlabel("Plays")
plt.ylabel("Accumulated success rate")
plt.title("10 Armed bandit with 3 base contexts / arm")
plt.xlim([0,no_events])
plt.ylim([0.0,0.10])
plt.legend(loc=2)
'''
    f2= plt.figure(2)
    plt.scatter(x,best_choice/no_runs,s=0.5,color=colors[cind],label=labels[cind])
    plt.xlim([0,no_events])
    plt.ylim([0.0,0.9])
    plt.legend(loc=2)
    plt.xlabel("Plays")
    plt.ylabel("%  of best arm played")
    plt.title("10 Armed bandit with parameters 0.01,...,0.1")

    f3= plt.figure(3)
    plt.scatter(x,spot_ctr/no_runs,s=0.5,color=colors[cind],label=labels[cind])
    plt.xlim([0,no_events])
    plt.ylim([0.0,0.10])
    plt.legend(loc=2)
    plt.xlabel("Plays")
    plt.ylabel("Non-accumulated success rate")
    plt.title("10 Armed bandit with parameters 0.1,...,1.0")

'''
f1.show()

#f2.show()

#f3.show()
raw_input()

plt.close()


'''
R=0
R_rand=0
no_events=1000
no_runs=1
alp=0.5
no_arms=10
band=Bandit(no_arms=no_arms)

linUCB_model=ucb.LINUCB(alp,10)
CTR_linUCB=np.zeros(no_events)

ucb1=UCB1(no_arms=no_arms)
CTR_ucb1=np.zeros(no_events)

for i in range(no_arms):
    linUCB_model.add_arm(i)


for j in range(no_runs):
    print "new_run"
    R_linUCB=0
    R_ucb1=0
    for i in range(no_events):
        features,ctx=band.context_gen()
        arm=linUCB_model.recommend(features)
        #rand_arm=random.randint(0,no_arms-1)
        fb=band.feed_back(arm,ctx)
        linUCB_model.update_stats(arm,features,fb)
        R_linUCB+=fb
        CTR_linUCB[i]+=R_linUCB/(i+1)

        arm2=ucb1.recommend()
        fb2=band.feed_back(arm2,ctx)
        ucb1.update_stats(arm2,fb2)
        R_ucb1+=fb2
        CTR_ucb1[i]+=R_ucb1/(i+1)

        fb3=band.feed_back(random.randint(0,no_arms-1),ctx)
        R_rand+=fb3
        if i%1000==0:
            print "reg UCB CTR: " + str(R_ucb1/(i+1))
            print "random  CTR: " + str(R_rand/(i+1))

'''
