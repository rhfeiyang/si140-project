import random
import math
import numpy
import copy

# super parameters
actual_theta = [0.7, 0.5, 0.4]
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
    I_t = -1

    for t in range(1, N + 1):
        if random.uniform(0, 1) < epsilon:
            I_t = arms[random.randint(0, len(arms) - 1)]
        else:
            I_t = 1
            for i in arms:
                if theta[i] > theta[I_t]:
                    I_t = i

        count[I_t] += 1
        theta[I_t] += (1 / count[I_t]) * (reward_part1(I_t) - theta[I_t])
    return theta[1:]


def Ucb(N, arms, c):
    # note the index start from 1
    I_t = -1
    count = [0 for i in range(len(arms) + 1)]
    theta = [0 for i in range(len(arms) + 1)]

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
        theta[I_t] += (reward_part1(I_t) - theta[I_t]) / count[I_t]
    return theta[1:]


def TS(N, arms, ab_original):
    ab = copy.deepcopy(ab_original)
    theta = [0 for i in range(len(arms) + 1)]
    for t in range(1, N + 1):
        # sample model
        for j in arms:
            theta[j] = numpy.random.beta(ab[j - 1][0], ab[j - 1][1])

        # select and pull arm
        I_t = -1
        arg_max = -1
        for j in arms:
            if theta[j] > arg_max:
                I_t = j
                arg_max = theta[j]

        # update distribution
        r = reward_part1(I_t)
        ab[I_t - 1][0] += r
        ab[I_t - 1][1] += (1 - r)
    # compute the expectation!!
    result = []
    for j in arms:
        result.append(ab[j - 1][0] / (ab[j - 1][0] + ab[j - 1][1]))
    return result


def result_part1(function_idx):
    function = ['epsilon-greedy', 'UCB', 'TS']
    print("results for", function[function_idx - 1], "Algorithm:")
    para = []
    func = None
    if function_idx == 1:
        para = GREEDY_epsilon
        func = e_Greedy
    elif function_idx == 2:
        para = UCB_c
        func = Ucb
    elif function_idx == 3:
        para = TS_ab
        func = TS

    for p in para:
        result = numpy.array([0.0, 0.0, 0.0])
        for trial in range(trial_times):
            result += numpy.array(func(N, arms_part1, p))
        result /= trial_times
        print(result, "with parameter:", p)


result_part1(1)
result_part1(2)
result_part1(3)
