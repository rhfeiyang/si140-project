import csv
import random
actual_theta = [0.7, 0.5, 0.4]
time=1000
def reward_part1(choose):
    # choose: 1,2,3
    probability = actual_theta[choose - 1]
    if (random.uniform(0, 1) < probability):
        return 1
    else:
        return 0

with open('dependent_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow( ['arm1']+['arm2']+['arm3'])
    for t in range(time):
        r1=reward_part1(1)
        r2 =reward_part1(2)
        r3 = reward_part1(3)
        writer.writerow([r1] + [r2] + [r3])

