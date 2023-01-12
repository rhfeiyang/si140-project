import random
import numpy as np

gamma=0.9
actual_theta = [0.7, 0.5]
N = 25
arms = [1,2]

test_round = 40
# EstimatedReward1 = [[0 for i in range(test_round)] for j in range(test_round)]
# EstimatedReward2 = [[0 for i in range(test_round)] for j in range(test_round)]
R_result=[[[[-1 for j4 in range(test_round)] for j3 in range(test_round)] for j2 in range(test_round)] for j1 in range(test_round)]
R_choose=[[[[-1 for j4 in range(test_round)] for j3 in range(test_round)] for j2 in range(test_round)] for j1 in range(test_round)]

def reward_part2(choose,time):
    # choose: 1,2
    assert choose>=0
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
        r = [R1(a1,b1,a2,b2),R2(a1,b1,a2,b2)]
        R=max(r[0],r[1])
        R_result[a1][b1][a2][b2]=R
        if(r[0]>r[1]):
            R_choose[a1][b1][a2][b2] = 1
        else:
            R_choose[a1][b1][a2][b2] = 2
        return R

def R1(a1, b1, a2, b2):
    c=a1/(a1+b1)
    return c*(1+gamma*R(a1+1,b1,a2,b2))+(1-c)*(gamma*R(a1,b1+1,a2,b2))

def R2(a1, b1, a2, b2):
    c=a2/(a2+b2)
    return c*(1+gamma*R(a1,b1,a2+1,b2))+(1-c)*(gamma*R(a1,b1,a2,b2+1))



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

        # print(I_t)
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

def R_part2(N):
    total_reward = 0.0
    a = [1,1]
    b = [1,1]
    for n in range(N):
        if R_choose[a[0]][b[0]][a[1]][b[1]]==-1:
            R(a[0],b[0],a[1],b[1])
        arm_choose = R_choose[a[0]][b[0]][a[1]][b[1]]
        r = reward_part2(arm_choose-1,n)
        if(r != 0):
            a[arm_choose-1] += 1
        else:
            b[arm_choose-1] += 1
        total_reward += r
    return total_reward

# print(R(1,1,1,1))
result1 = 0
result2 = 0
for n in range(100):
    result1 += Part2(N,arms)
    result2 += R_part2(N)
print(result1/100)
print(result2/100)