import random
import math
import copy
import numpy as np

actual_theta = [0.7, 0.5, 0.4]
N = 5000
trial_times = 200
arms_part1 = [1, 2, 3]

GREEDY_epsilon = [0.1, 0.5, 0.9]
UCB_c = [1, 5, 10]
TS_ab = [[[1, 1], [1, 1], [1, 1]],
         [[601, 401], [401, 601], [2, 3]]]


def reward_part1(choose):
    # choose: 1,2,3
    probability = actual_theta[choose - 1]
    if (random.uniform(0, 1) < probability):
        return 1
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
        # I_t=1,2,3...
        if random.uniform(0, 1) < epsilon:
            I_t = arms[random.randint(0, len(arms) - 1)]
        else:
            I_t = np.argmax(theta)
            if I_t == 0:
                I_t = 1

        count[I_t] += 1
        r = reward_part1(I_t)
        total_reward += r
        theta[I_t] += (1 / count[I_t]) * (r - theta[I_t])
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
        theta[I_t] = reward_part1(I_t)

    for t in range(4, N + 1):
        # select and pull arm

        I_t = 1
        arg_max = 0
        for j in arms:
            arg = theta[j] + c * math.sqrt(2 * math.log(t) / count[j])
            if arg > arg_max:
                I_t = j
                arg_max = arg
        count[I_t] += 1
        # print(I_t)
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

    total_reward = 0
    #ab idx start from 0
    ab = copy.deepcopy(ab_original)
    for t in range(1, N + 1):
        I_t = TS_arm_choose(ab)
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


#TODO
def depend_TS(N, arms, ab_original):
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
        arm_choose = TS_arm_choose([ab[i - 1] for i in cluster_set[cluster_choose]])
        print('cluster:', cluster_theta)
        # print(arm_choose)
        # update distribution
        r = reward_part1(arm_choose)
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

def result_part1(function_idx):
    function = ['epsilon-greedy', 'UCB', 'TS', 'TS-D', 'TS-DD']
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
        para = TS_ab
        func = depend_TS
        parameter = "a,b ,cluster:[[1,2],[3]]"
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
# result_part1(2)
result_part1(3)
# result_part1(4)
# result_part1(5)