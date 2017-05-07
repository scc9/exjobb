from __future__ import division
import numpy as np
import random
import scipy as sp
import matplotlib.pyplot as plt

'''
Multi armed bandit test bench with support for multi contexts

'''


class Bandit():

    def __init__(self,no_arms=5):
        #List of list of lists
        self.bern_p=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
        self.no_arms=no_arms

    def feed_back(self,arm):
        return random.random()<=self.bern_p[arm]
class HyperTS():
    def __init__(self,arm_list,alpha=0.5,beta=0.5):
        self.no_arms=len(arm_list)
        self.beta=beta
        self.curr_arm=0
        self.alpha=alpha
        self.no_success=np.ones(self.no_arms)*alpha
        self.no_fails=np.ones(self.no_arms)*beta
        self.arm_list=arm_list

    def recommend(self):
        p=[np.random.beta(self.no_success[arm],self.no_fails[arm]) for arm in range(self.no_arms)]
        arm=np.argmax(p)
        self.curr_arm=arm
        return self.arm_list[arm].recommend()

    def update_stats(self,arm,feedback):
        self.no_fails[self.curr_arm]=self.no_fails[self.curr_arm]*0.9+ (1-feedback)*10
        self.no_success[self.curr_arm]=self.no_success[self.curr_arm]*0.9 + feedback*10
        for alg in self.arm_list:
            alg.update_stats(arm,feedback)

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

    def recommend(self):
        p=[np.random.beta(self.no_success[arm],self.no_fails[arm]) for arm in self.arm_list]
        return np.argmax(p)

    def update_stats(self,arm,feedback):
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

    def recommend(self):
        if random.random()<=self.epsilon:
            return random.choice(self.arm_list)
        else:
            return np.argmax(self.mean)

    def update_stats(self,arm,feedback):
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

    def recommend(self):
        #p=[self.ucb(arm) for arm in self.arm_list]
        return np.argmax(self.p)

    def update_stats(self,arm,feedback):
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


no_events=1000
no_runs=10000

band=Bandit(no_arms=10)
hyperTS=HyperTS([Thompson(no_arms=10,alpha=0.5,beta=0.5),UCB1(no_arms=10),E_greedy(no_arms=10,epsilon=0.1)])
hyperUCB1=HyperUCB1([Thompson(no_arms=10,alpha=0.5,beta=0.5),UCB1(no_arms=10),E_greedy(no_arms=10,epsilon=0.1)])
thomp=Thompson(no_arms=10,alpha=0.5,beta=0.5)
ucb1=UCB1(no_arms=10)
e_greedy1=E_greedy(no_arms=10,epsilon=0.1)
#avg_ctr=np.zeros(1000)
models=[hyperUCB1,e_greedy1,ucb1,thomp]
colors=['black','blue','red','green']
labels=['hyperUCB1','$\epsilon$-greedy 0.1', "Thompson", "UCB1"]

cind=0
stats_mtrx=np.empty([no_runs,no_events])


for model in models:
    print "New Model"
    best_choice=np.zeros(no_events)
    spot_ctr=np.zeros(no_events)
    for j in range(no_runs):
        print j
        CTR=[]
        t=0
        r=0

        for i in range(no_events):
            arm=model.recommend()
            fb=band.feed_back(arm)
            model.update_stats(arm,fb)

            #arm2=e_greedy.recommend()
            #fb2=band.feed_back(arm2)
            #e_greedy2.update_stats(arm2,fb2)

            #arm=thomp.recommend()
            #fb=band.feed_back(arm)
            #thomp.update_stats(arm,fb)

            #arm=ucb1.recommend()
            #fb=band.feed_back(arm)
            #ucb1.update_stats(arm,fb)
            best_choice[i]+=arm==9
            spot_ctr[i]+= (arm+1)/100
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
        std[k]=np.std(stats_mtrx[:,k])/2  #half std
    x=range(no_events)
    up=np.add(mean,std/2)
    down=np.subtract(mean,std/2)
    f1 = plt.figure(1)
    plt.plot(x,mean,linewidth=3.0,color=colors[cind],label=labels[cind])
    #plt.fill_between(x,up,down,facecolor=colors[cind],alpha=0.3)
    plt.xlabel("Plays")
    plt.ylabel("Accumulated success rate")
    plt.title("10 Armed bandit with parameters 0.1,...,1.0")
    plt.xlim([0,no_events])
    plt.ylim([0.0,0.10])
    plt.legend(loc=2)
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
    cind+=1

f1.show()

f2.show()

f3.show()
raw_input()

plt.close()
