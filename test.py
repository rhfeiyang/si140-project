import random
import math
import numpy
import copy

#super parameters
actual_theta=[0.7,0.5,0.4]
N=5000
trial_times=200
arms_part1=[1,2,3]
# arms_part2=[1,2]

GREEDY_epsilon=[0.1,0.5,0.9]
UCB_c=[1,5,10]
TS_ab=[[[1,1],[1,1],[1,1]],
       [[601,401],[401,601],[2,3]]]

def reward_part1(choose):
    #choose: 1,2,3
    probability=actual_theta[choose-1]
    if(random.uniform(0,1)<probability):
        return 1
    else:
        return 0



def TS(N, arms, ab_original):
    ab = copy.deepcopy(ab_original)
    theta = [0 for i in range(len(arms) + 1)]
    for t in range(1, N + 1):
        # sample model
        for j in arms:
            theta[j] = numpy.random.beta(ab[j-1][0], ab[j-1][1])
        I_t = 1
        arg_max = 0

        # select and pull arm
        for j in arms:
            if theta[j] > arg_max:
                I_t = j
                arg_max = theta[j]

        # update distribution
        r = reward_part1(I_t)
        ab[I_t-1][0] += r
        ab[I_t-1][1] += 1 - r
    result=[]
    for j in arms:
        result.append(ab[j-1][0]/(ab[j-1][0]+ab[j-1][1]))
    return result

for ab in TS_ab:
    result = numpy.array([0.0, 0.0, 0.0])
    for trial in range(trial_times):
        result += numpy.array(TS(N, arms_part1, ab))
    result/=trial_times
    print(result)
