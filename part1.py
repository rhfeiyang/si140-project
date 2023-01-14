import operator
import random
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_loader
import cython.parallel as parallel
actual_theta = [0.7, 0.5, 0.4]
N = 5000
trial_times = 1
arms_part1 = [1, 2, 3]

GREEDY_epsilon = [0.1, 0.5, 0.9]
UCB_c = [1, 5, 10]
TS_ab = [[[1, 1], [1, 1], [1, 1]],
         [[601, 401], [401, 601], [2, 3]]]

greedy_regret = np.zeros(N+1)
TS_regret = np.zeros(N+1)
UCB_regret = np.zeros(N+1)
dependent_UCB_regret = np.zeros(N+1)

class P1:

    def __init__(self, depend_data_path=None):
        self.path = depend_data_path
        if depend_data_path is not None:
            self.loader=data_loader.Loader(depend_data_path)
            self.actual_theta=self.loader.means
            self.optArm=self.loader.optArm
            self.arm_num=self.loader.arm_num
            self.sampler=self.loader.sample
        else:
            self.loader=None
            self.actual_theta=actual_theta
            self.optArm=np.argmax(self.actual_theta)
            self.arm_num = len(self.actual_theta)
            self.sampler=self.independ_reward

    def independ_reward(self, choose):
        # choose: 1,2,3
        probability = self.actual_theta[choose]
        if random.uniform(0, 1) < probability:
            return 1
        else:
            return 0

    def e_Greedy(self, N, epsilon):
        # initialize
        # Notice we set index start from 1
        theta = [0]*self.arm_num
        count =[0]*self.arm_num
        total_reward = 0
        greedy_current_regret = np.zeros(N+1)

        for t in range(1, N + 1):
            # I_t=1,2,3...
            if random.uniform(0, 1) < epsilon:
                I_t = random.randint(0, self.arm_num-1)
            else:
                I_t = np.argmax(theta)
                if I_t == 0:
                    I_t = 1

            count[I_t] += 1
            r = self.sampler(I_t)
            total_reward += r
            theta[I_t] += (1 / count[I_t]) * (r - theta[I_t])
            if t==0:
                greedy_current_regret[t] = self.actual_theta[self.optArm] - self.actual_theta[I_t]
            else:
                greedy_current_regret[t] = greedy_current_regret[t-1] + self.actual_theta[self.optArm] - self.actual_theta[I_t]
        global greedy_regret
        greedy_regret += greedy_current_regret
        return total_reward

    def Ucb(self, N, c):
        # note the index start from 0
        count = [0]* self.arm_num
        theta = [0]* self.arm_num
        total_reward = 0
        UCB_current_regret = np.zeros(N+1)

        # initialize
        for t in range(self.arm_num):
            I_t = t
            count[I_t] = 1
            theta[I_t] = self.independ_reward(I_t)
            if t==0:
                UCB_current_regret[t] = actual_theta[self.optArm] - actual_theta[I_t]
            else:
                UCB_current_regret[t] = UCB_current_regret[t-1] + actual_theta[self.optArm] - actual_theta[I_t]

        for t in range(self.arm_num, N + 1):
            # select and pull arm
            I_t =np.argmax([theta[j] + c * math.sqrt(2 * math.log(t) / count[j]) for j in range(self.arm_num)])

            count[I_t] += 1
            r = self.sampler(I_t)
            total_reward += r
            theta[I_t] += (r - theta[I_t]) / count[I_t]
            UCB_current_regret[t] = UCB_current_regret[t - 1] + self.actual_theta[self.optArm] - self.actual_theta[I_t]
        global UCB_regret
        UCB_regret += UCB_current_regret
        return total_reward

    def depend_Ucb(self, N, c):
        table=self.loader.table
        arm_num=self.loader.arm_num

        count = np.array([0]*arm_num)
        theta = np.array([0.]*arm_num)
        ucb_idx=dict(zip(range(arm_num),[np.inf]*arm_num))
        ave_pseudo_reward=np.array([[np.inf]*arm_num]*arm_num)
        sum_pseudo_reward=np.array([[0.]*arm_num]*arm_num)
        d_UCB_current_regret = np.zeros(N+1)

        total_reward = 0
        for t in range(N):
            if t<arm_num:
                choose=t
            else:
                S_bool=(count>=(float(t-1)/arm_num))
                k_emp_reward=np.max(theta[S_bool])
                k_emp=np.where(theta==k_emp_reward)[0][0]
                comp_set=set()
                comp_set.add(k_emp)
                min_phi = np.min(ave_pseudo_reward[:, S_bool], axis=1)
                for k in range(arm_num):
                    if min_phi[k]>= k_emp_reward:
                        comp_set.add(k)

                comp_idx={ind: ucb_idx[ind] for ind in comp_set}
                choose=max(comp_idx.items(),key=operator.itemgetter(1))[0]
            # print(t,choose)
            if t==0:
                d_UCB_current_regret[t+1] = self.actual_theta[self.optArm] - self.actual_theta[choose]
            else:
                d_UCB_current_regret[t+1] = d_UCB_current_regret[t] + self.actual_theta[self.optArm] - self.actual_theta[choose]

            reward=self.sampler(choose)
            count[choose]+=1
            theta[choose]+=((reward-theta[choose])/count[choose])

            for arm in range(arm_num):
                if (count[arm] > 0):
                    ucb_idx[arm] = theta[arm] + c * np.sqrt(2 * np.log(t + 1) / count[arm])

            # pseudoReward=table[choose][reward]
            pseudoReward = table[choose][reward - 1,:]
            sum_pseudo_reward[:, choose] = sum_pseudo_reward[:, choose]+ pseudoReward
            ave_pseudo_reward[:, choose] = np.divide(sum_pseudo_reward[:, choose], count[choose])

            ave_pseudo_reward[np.arange(arm_num),np.arange(arm_num)]=theta

            total_reward+=reward
        global dependent_UCB_regret
        dependent_UCB_regret += d_UCB_current_regret
        return total_reward

    def TS_arm_choose(self, ab):
        theta = [0 for i in ab]
        for i, (a, b) in enumerate(ab):
            theta[i] = np.random.beta(a, b)
        return np.argmax(theta)

    def TS(self, N, ab_original):

        total_reward = 0
        # ab idx start from 0
        ab = copy.deepcopy(ab_original)
        TS_current_regret = np.zeros(N+1)
        for t in range(1, N + 1):
            I_t = self.TS_arm_choose(ab)
            # print(I_t)
            # update distribution
            r = self.sampler(I_t)
            ab[I_t - 1][0] += r
            ab[I_t - 1][1] += (1 - r)
            total_reward += r
            if t==0:
                TS_current_regret[t] = self.actual_theta[self.optArm] - self.actual_theta[I_t]
            else:
                TS_current_regret[t] = TS_current_regret[t-1] + self.actual_theta[self.optArm] - self.actual_theta[I_t]
        # compute the expectation!!
        # result = []
        # for j in arms:
        #     result.append(ab[j - 1][0] / (ab[j - 1][0] + ab[j - 1][1]))
        # print(choose_first_cluster)
        global TS_regret
        TS_regret += TS_current_regret
        return total_reward

    # TODO
    def depend_TS(self, N, arms, ab_original):
        total_reward = 0
        choose_first_cluster = 0
        ab = copy.deepcopy(ab_original)

        # use ucb for clusters
        # try: one cluster for 1,2 another for 3

        # cluster_ab=[[ab[0][0]+ab[1][0],ab[0][1]+ab[1][1]],[ab[2][0],ab[2][1]]]

        cluster_set = [[1, 2], [3]]
        cluster_count = []

        cluster_theta = np.array([0.0, 0.0])
        c = 5

        for i, cluster in enumerate(cluster_set):
            cluster_count.append(sum([sum(ab[arm - 1]) for arm in cluster]))
            cluster_theta[i] = float(sum([ab[arm - 1][0] for arm in cluster])) / cluster_count[i]

        cluster_count = np.array(cluster_count)

        for t in range(1, N + 1):
            # select cluster
            cluster_choose = np.argmax(cluster_theta + c * np.sqrt(2 * math.log(t) / cluster_count))

            # select and pull arm
            arm_choose = self.TS_arm_choose([ab[i - 1] for i in cluster_set[cluster_choose]])
            print('cluster:', cluster_theta)
            # print(arm_choose)
            # update distribution
            r = self.independ_reward(arm_choose)
            ab[arm_choose - 1][0] += r
            ab[arm_choose - 1][1] += (1 - r)
            cluster_count[cluster_choose] += 1
            cluster_theta[cluster_choose] += 1 / (cluster_count[cluster_choose]) * (r - cluster_theta[cluster_choose])

            total_reward += r
        # compute the expectation!!
        # result = []
        # for j in arms:
        #     result.append(ab[j - 1][0] / (ab[j - 1][0] + ab[j - 1][1]))
        # print(choose_first_cluster)
        return total_reward



    # def depend_TS_2(N, arms, ab_original):
    #     total_reward = 0
    #     ab = copy.deepcopy(ab_original)
    #     theta = [0 for i in range(len(arms) + 1)]
    #     # try: one cluster for 1,2 another for 3
    #     cluster = [[1, 2], [3]]
    #     cluster_theta = [ab[0][0] / (2 * (ab[0][0] + ab[0][1])) + ab[1][0] / (2 * (ab[1][0] + ab[1][1])),
    #                      ab[2][0] / (ab[2][0] + ab[2][1])]
    #
    #     for t in range(1, N + 1):
    #
    #         cluster_theta = [ab[0][0] / (2 * (ab[0][0] + ab[0][1])) + ab[1][0] / (2 * (ab[1][0] + ab[1][1])),
    #                          ab[2][0] / (ab[2][0] + ab[2][1])]
    #
    #         for j in arms:
    #             theta[j] = numpy.random.beta(ab[j - 1][0], ab[j - 1][1])
    #
    #         # select cluster
    #         cluster_choose = -1
    #         arg_max = -1
    #         for i, j in enumerate(cluster_theta):
    #             if j > arg_max:
    #                 arg_max = j
    #                 cluster_choose = i
    #
    #         # select and pull arm
    #         arm_choose = -1
    #         arg_max = -1
    #         for i in cluster[cluster_choose]:
    #             if theta[i] > arg_max:
    #                 arm_choose = i
    #                 arg_max = theta[i]
    #
    #         # update distribution
    #         r = reward_part1(arm_choose)
    #         if (arm_choose == 1):
    #             ab[0][0] += r
    #             ab[0][1] += (1 - r)
    #             ab[1][0] += r
    #             ab[1][1] += (1 - r)
    #         elif (arm_choose == 2):
    #             ab[0][0] += r
    #             ab[0][1] += (1 - r)
    #             ab[1][0] += r
    #             ab[1][1] += (1 - r)
    #         elif (arm_choose == 3):
    #             ab[2][0] += r
    #             ab[2][1] += (1 - r)
    #
    #         total_reward += r
    #     # compute the expectation!!
    #     # result = []
    #     # for j in arms:
    #     #     result.append(ab[j - 1][0] / (ab[j - 1][0] + ab[j - 1][1]))
    #     return total_reward

    def result(self, function_idx):
        function = ['epsilon-greedy', 'UCB', 'TS', 'D-UCB']
        print("results for", function[function_idx - 1], "Algorithm:")
        para = []
        func = None
        if function_idx == 1:
            para = GREEDY_epsilon
            func = self.e_Greedy
            parameter = "epsilon"
        elif function_idx == 2:
            para = UCB_c
            func = self.Ucb
            parameter = "c"
        elif function_idx == 3:
            para = TS_ab
            func = self.TS
            parameter = "a,b"
        elif function_idx == 4:
            para = UCB_c
            func = self.depend_Ucb
            parameter = "c"
        # elif function_idx == 5:
        #     para = TS_ab
        #     func = depend_TS_2
        #     parameter = "a,b ,cluster:[[1,2],[3]]"

        for p in para:
            result = 0.0
            for trial in parallel.prange(trial_times):
                result += func(N, p)
            result /= trial_times
            print(result, "with parameter", parameter, "as", p)


p1 = P1('movie_3.csv')
# p1.result(1)
p1.result(2)
# p1.result(3)
p1.result(4)


# proccessing data: taking the mean of regrets
greedy_regret /= trial_times
TS_regret /= trial_times
UCB_regret /= trial_times
dependent_UCB_regret /= trial_times

# plot
spacing = int(N/20)
plt.plot(range(0, N+1)[::spacing], greedy_regret[::spacing], label='epsilon-Greedy', color='black', marker='x')
plt.plot(range(0, N+1)[::spacing], UCB_regret[::spacing], label='UCB', color='red', marker='+')
plt.plot(range(0, N+1)[::spacing], TS_regret[::spacing], label='TS', color='yellow', marker='o')
plt.plot(range(0, N+1)[::spacing], dependent_UCB_regret[::spacing], label='D-UCB', color='blue', marker='^')
plt.legend()
plt.grid(True, axis='y')
plt.xlabel('Number of Rounds')
plt.ylabel('Average Regret')
plt.show()