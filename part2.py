import random

gamma=0.9
actual_theta = [0.7, 0.5]

test_round = 50
# EstimatedReward1 = [[0 for i in range(test_round)] for j in range(test_round)]
# EstimatedReward2 = [[0 for i in range(test_round)] for j in range(test_round)]
R_result=[[[[-1 for j4 in range(test_round)] for j3 in range(test_round)] for j2 in range(test_round)] for j1 in range(test_round)]

def reward_part2(choose,time):
    # choose: 1,2
    probability = actual_theta[choose]
    if (random.uniform(0, 1) < probability):
        return gamma**time
    else:
        return 0

def R(a1, b1, a2, b2):
    if(a1>=test_round or a2>=test_round or b1 >= test_round or b2>=test_round):
        return 0
    if R_result[a1][b1][a2][b2]!=-1:
        return R_result[a1][b1][a2][b2]
    else:
        R=max(R1(a1,b1,a2,b2), R2(a1,b1,a2,b2))
        R_result[a1][b1][a2][b2]=R
        return R

def R1(a1, b1, a2, b2):
    c=a1/(a1+b1)
    return c*(1+gamma*R(a1+1,b1,a2,b2))+(1-c)*(gamma*R(a1,b1+1,a2,b2))

def R2(a1, b1, a2, b2):
    c=a2/(a2+b2)
    return c*(1+gamma*R(a1,b1,a2+1,b2))+(1-c)*(gamma*R(a1,b1,a2,b2+1))

# def R1(a1, b1, a2, b2):
#     if EstimatedReward1[a1][b1] != 0:
#         return EstimatedReward1[a1][b1]
#     if a1==test_round-1 and b1==test_round-1:
#         EstimatedReward1[a1][b1] = 1/2
#     elif a1==test_round-1:
#         r = R(a1, b1 + 1, a2, b2)
#         EstimatedReward1[a1][b1] = a1/(a1+b1) + (b1*gamma*r)/(a1+b1)
#     elif b1==test_round-1:
#         r = R(a1 + 1, b1, a2, b2)
#         EstimatedReward1[a1][b1] = (a1*(1+gamma*r))/(a1+b1)
#     else:
#         r1 = R(a1, b1 + 1, a2, b2)
#         r2 = R(a1 + 1, b1, a2, b2)
#         EstimatedReward1[a1][b1] = (a1*(1+gamma*r2))/(a1+b1) + (b1*gamma*r1)/(a1+b1)
#     return EstimatedReward1[a1][b1]
#
# def R2(a1, b1, a2, b2):
#     if EstimatedReward1[a2][b2] != 0:
#         return EstimatedReward1[a2][b2]
#     if a2==test_round-1 and b2==test_round-1:
#         EstimatedReward2[a2][b2] = 1/2
#     elif a2==test_round-1:
#         r = R(a1, b1, a2, b2 + 1)
#         EstimatedReward2[a2][b2] = a2/(a2+b2) + (b2*gamma*r)/(a2+b2)
#     elif b2==test_round-1:
#         r = R(a1, b1, a2 + 1, b2)
#         EstimatedReward2[a2][b2] = (a2*(1+gamma*r))/(a2+b2)
#     else:
#         r1 = R(a1, b1, a2, b2 + 1)
#         r2 = R(a1, b1, a2 + 1, b2)
#         EstimatedReward2[a2][b2] = (a2*(1+gamma*r2))/(a2+b2) + (b2*gamma*r1)/(a2+b2)
#     return EstimatedReward1[a2][b2]



def Part2(N,arms):
    #initialize:

    # I_t = -1
    #idx here all start from 0
    count = [0 for i in range(len(arms))]
    theta = [0.5 for i in range(len(arms))]
    ab = [[1,1] for arm in arms]

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

print(R(1,1,1,1))
