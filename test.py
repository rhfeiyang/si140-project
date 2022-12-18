import random
import math
import numpy
import copy

#super parameters
#actual_theta=[0.7,0.7,0.4]
actual_theta=[0.7,0.3]
gamma = 0.9;
N=5000
trial_times=200
#arms_part1=[1,2,3]
arms_part2=[1,2]

GREEDY_epsilon=[0.1,0.5,0.9]
UCB_c=[1,5,10]
TS_ab=[[[1,1],[1,1],[1,1]],
       [[601,401],[401,601],[2,3]]]

# def reward_part1(choose):
#     #choose: 1,2,3
#     probability=actual_theta[choose-1]
#     if(random.uniform(0,1)<probability):
#         return 1
#     else:
#         return 0

def reward_part2(choose,time):
    # choose: 1,2
    probability = actual_theta[choose - 1]
    if (random.uniform(0, 1) < probability):
        return gamma**time
    else:
        return 0

def Part2(N,arms):
    #initialize:
    I_t = -1
    count = [0 for i in range(len(arms) + 1)]
    theta = [0 for i in range(len(arms) + 1)]
    total_reward = 0

# def Ucb(N, arms, c):
#     # note the index start from 1
#     I_t = -1
#     count = [0 for i in range(len(arms) + 1)]
#     theta = [0 for i in range(len(arms) + 1)]
#     total_reward = 0
#
#     # initialize
#     for t in arms:
#         I_t = t
#         count[I_t] = 1
#         theta[I_t] = reward_part1(I_t)
#
#     for t in range(4, N + 1):
#         # select and pull arm
#
#         I_t = 1
#         arg_max = 0
#         for j in arms:
#             arg = theta[j] + c * math.sqrt(2 * math.log(t) / count[j])
#             if arg > arg_max:
#                 I_t = j
#                 arg_max = arg
#         count[I_t] += 1
#         r = reward_part1(I_t)
#         total_reward += r
#         theta[I_t] += (r - theta[I_t]) / count[I_t]
#         # theta[I_t] += (reward_part1(I_t) - theta[I_t]) / count[I_t]
#         #print(I_t)
#     return total_reward
#
# def Ucb_D(N, arms, c):
#     I_t = -1
#     count = [0 for i in range(len(arms) + 1)]
#     theta = [0 for i in range(len(arms) + 1)]
#     cluster_set = [[1, 2], [3]]
#     cluster_theta = [0,0]
#     total_reward = 0
#     for t in arms:
#         I_t = t
#         count[I_t] = 1
#         theta[I_t] = reward_part1(I_t)
#     #if in cluster [1,2], 1 has reward probability of theta, 2 has theta
#     for t in range(4, N + 1):
#         # select cluster
#         C_t = 1;
#         cluster_theta[0] = max(theta[0],theta[1])
#         cluster_theta[1] = theta[2]
#         if(cluster_theta[0]>cluster_theta[1]):
#             C_t = 0;
#
#         #select and pull arm
#         I_t = 1
#         arg_max = 0
#         for j in cluster_set[C_t]:
#             arg = theta[j] + c * math.sqrt(2 * math.log(t) / count[j])
#             if arg > arg_max:
#                 I_t = j
#                 arg_max = arg
#         count[I_t] += 1
#         r = reward_part1(I_t)
#         total_reward += r
#         theta[I_t] += (r - theta[I_t]) / count[I_t]
#         if(I_t ==1):
#             theta[0] += (r - theta[I_t]) / count[I_t]
#         if (I_t == 0):
#             theta[1] += (r - theta[I_t]) / count[I_t]
#         # theta[I_t] += (reward_part1(I_t) - theta[I_t]) / count[I_t]
#         #print(I_t)
#     return total_reward
#
#
# for c in UCB_c:
#     result = 0.0
#     for trial in range(trial_times):
#         result += Ucb(N, arms_part1, c)
#     result/=trial_times
#     print(result)
#
# for c in UCB_c:
#     result = 0.0
#     for trial in range(trial_times):
#         result += Ucb_D(N, arms_part1, c)
#     result/=trial_times
#     print(result)
