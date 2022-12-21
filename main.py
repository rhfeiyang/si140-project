import random
import math
# import numpy
import copy

# super parameters
import numpy as np

actual_theta = [0.4,0.4,0.6]
# actual_theta = [0.6,0.6,0.4]
N = 5000
trial_times = 200
arms_part1 = [1, 2, 3]
# arms_part2=[1,2]

GREEDY_epsilon = [0.1, 0.5, 0.9]
UCB_c = [1, 5, 10]
TS_ab = [[[1, 1], [1, 1], [1, 1]],
         [[601, 401], [401, 601], [2, 3]]]


def reward_part1(choose):
    # choose: 1,2,3
    if(choose!=1 and choose !=2 and choose!=3):
        print(choose,"wrong")
    probability = actual_theta[choose - 1]
    if (random.uniform(0, 1) < probability):
        return 1
    else:
        return 0

def reward_part2(choose,time):
    # choose: 1,2
    probability = actual_theta[choose]
    if (random.uniform(0, 1) < probability):
        return gamma**time
    else:
        return 0

def e_Greedy(N, arms, epsilon):
    # initialize
    # Notice we set index start from 1
    theta = [0 for i in range(len(arms) + 1)]
    count = [0 for i in range(len(arms) + 1)]
    total_reward = 0
    I_t = -1

    for t in range(1, N + 1):
        if random.uniform(0, 1) < epsilon:
            I_t = arms[random.randint(0, len(arms) - 1)]
        else:
            I_t=np.argmax(theta)
            if I_t==0:
                I_t=1
            # I_t = 1
            # for i in arms:
            #     if theta[i] > theta[I_t]:
            #         I_t = i

        count[I_t] += 1
        r = reward_part1(I_t)
        total_reward += r
        theta[I_t] += (1 / count[I_t]) * (r - theta[I_t])
        # theta[I_t] += (1 / count[I_t]) * (reward_part1(I_t) - theta[I_t])
    return total_reward


def Ucb(N, arms, c):
    # note the index start from 1
    I_t = -1
    count = [0 for i in range(len(arms) + 1)]
    theta = [0 for i in range(len(arms) + 1)]
    total_reward = 0

    # initialize
    for t in arms:
        I_t = t
        count[I_t] = 1
        r=reward_part1(I_t)
        theta[I_t]=r
        total_reward+=r

    for t in range(4, N + 1):
        # select and pull arm

        I_t = -1
        arg_max = -1
        for j in arms:
            arg = theta[j] + c * math.sqrt(2 * math.log(t) / count[j])
            if arg > arg_max:
                I_t = j
                arg_max = arg
        count[I_t] += 1
        #print(I_t)
        r = reward_part1(I_t)
        total_reward += r
        theta[I_t] += (r - theta[I_t]) / count[I_t]

        # theta[I_t] += (reward_part1(I_t) - theta[I_t]) / count[I_t]
    return total_reward

def TS_arm_choose(ab):
    theta = [0 for i in ab]
    for i, (a, b) in enumerate(ab):
        theta[i] = np.random.beta(a, b)
    return np.argmax(theta) + 1

def TS(N, arms, ab_original):
    choose_first_cluster = 0
    total_reward = 0
    ab = copy.deepcopy(ab_original)
    for t in range(1, N + 1):
        # # sample model
        # for j in arms:
        #     theta[j] = numpy.random.beta(ab[j - 1][0], ab[j - 1][1])
        #
        # # select and pull arm
        # I_t = -1
        # arg_max = -1
        # for j in arms:
        #     if theta[j] > arg_max:
        #         I_t = j
        #         arg_max = theta[j]
        I_t=TS_arm_choose(ab)
        # print(I_t)
        # update distribution
        r = reward_part1(I_t)
        ab[I_t - 1][0] += r
        ab[I_t - 1][1] += (1 - r)
        total_reward += r
    # compute the expectation!!
    # result = []
    # for j in arms:
    #     result.append(ab[j - 1][0] / (ab[j - 1][0] + ab[j - 1][1]))
    # print(choose_first_cluster)
    return total_reward


def depend_UCB(N, arms, c):
    total_reward = 0
    choose_first_cluster = 0

    # use ucb for clusters
    # try: one cluster for 1,2 another for 3

    # cluster_ab=[[ab[0][0]+ab[1][0],ab[0][1]+ab[1][1]],[ab[2][0],ab[2][1]]]
    
    arm_theta = np.array([0.0,0.0,0.0])
    arm_count = np.array([1,1,1])


    cluster_set = [[1, 2], [3]]
    cluster_count = np.array([0,0])

    cluster_theta = np.array([0.0, 0.0])
    # c = 5

#     for i, cluster in enumerate(cluster_set):
#         cluster_count.append(sum([sum(ab[arm-1]) for arm in cluster]))
#         cluster_theta[i]=float(sum([ab[arm-1][0] for arm in cluster]))/cluster_count[i]

    for t in arms:
        I_t = t
        r=reward_part1(I_t)
        arm_theta[I_t-1] = r
        total_reward+=r

    for i, cluster in enumerate(cluster_set):
        cluster_count[i]+=len(cluster)
        # cluster_count[i] += 1
        # cluster_theta[i]+=sum([arm_theta[arm-1] for arm in cluster])/cluster_count[i]
        cluster_theta[i] = max([arm_theta[arm - 1] for arm in cluster])

    for t in range(4,N+1):
        # select cluster
        # cluster_choose = np.argmax(cluster_theta + c * np.sqrt(2 * math.log(t) / (cluster_count/np.array([len(i) for i in cluster_set]))))
        cluster_choose = np.argmax(
            cluster_theta + c * np.sqrt(2 * math.log(t) / cluster_count))
        # print(cluster_choose)
        # cluster_choose = np.argmax(cluster_theta)
        # select and pull arm
        arm_choose = -1
        arg_max = -1
        for j in cluster_set[cluster_choose]:
            arg = arm_theta[j-1] + c * math.sqrt(2 * math.log(t) / arm_count[j-1])
            if arg > arg_max:
                arm_choose = j-1
                arg_max = arg
        #print('cluster:', cluster_theta)
        # print(arm_choose)
        # update distribution
        # print(arm_choose)
        r = reward_part1(arm_choose+1)
        arm_count[arm_choose] += 1
        arm_theta[arm_choose] += 1 / (arm_count[arm_choose]) * (r - arm_theta[arm_choose])
        cluster_count[cluster_choose] += 1
        # cluster_theta[cluster_choose] += 1 / (cluster_count[cluster_choose]) * (r - cluster_theta[cluster_choose])

        for i, cluster in enumerate(cluster_set):
            cluster_theta[i] = max([arm_theta[arm - 1] for arm in cluster])

        total_reward += r

    # compute the expectation!!
    # result = []
    # for j in arms:
    #     result.append(ab[j - 1][0] / (ab[j - 1][0] + ab[j - 1][1]))
    # print(choose_first_cluster)
    return total_reward



def Part2(N,arms):
    #initialize:

    # I_t = -1
    count = [0 for i in range(len(arms))]
    theta = [0.5 for i in range(len(arms))]
    ab = [[1,1] for arm in arms]
    # for t in arms:
    #     theta[t] = np.random.beta(ab[t-1][0],ab[t-1][1])
    total_reward = 0
    for t in range(N):
        #choose and pull arm

        if(theta[0]==theta[1]):
            I_t=random.randint(0,1)
        else:
            I_t = np.argmax(theta)

        print(I_t)
        r = reward_part2(I_t,t)
        total_reward+=r
        count[I_t] +=1
        if(r!= 0):
            ab[I_t][0]+=1
        else:
            ab[I_t][1]+=1
        #更新theta
        theta = [float(a) / (a + b) for a, b in ab]
        # for t in arms:
        #     theta[t] = ab[I_t][0]/(ab[I_t][0]+ab[I_t][1])
    return total_reward


def result_part1(function_idx):
    function = ['epsilon-greedy', 'UCB', 'TS', 'UCB-D', 'TS-DD']
    print("results for", function[function_idx - 1], "Algorithm:")
    para = []
    func = None
    if function_idx == 1:
        para = GREEDY_epsilon
        func = e_Greedy
        parameter = "epsilon"
    elif function_idx == 2:
        para = UCB_c
        func = Ucb
        parameter = "c"
    elif function_idx == 3:
        para = TS_ab
        func = TS
        parameter = "a,b"
    elif function_idx == 4:
        para = UCB_c
        func = depend_UCB
        parameter = "c ,cluster:[[1,2],[3]]"
    # elif function_idx == 5:
    #     para = TS_ab
    #     func = depend_TS_2
    #     parameter = "a,b ,cluster:[[1,2],[3]]"

    for p in para:
        result = 0.0
        for trial in range(trial_times):
            result += func(N, arms_part1, p)
        result /= trial_times
        print(result, "with parameter", parameter, "as", p)


# result_part1(1)
result_part1(2)
# result_part1(3)
result_part1(4)
# result_part1(5)