import csv
import random
actual_theta = [0.7, 0.5, 0.4]
def data_generator(file_name):
    time = 10000
    def reward(choose):
        # choose: 1,2,3
        probability = actual_theta[choose - 1]
        if (random.uniform(0, 1) < probability):
            return 1
        else:
            return 0

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow( ['arm1']+['arm2']+['arm3'])
        for t in range(time):
            r1 = reward(1)
            r2 = reward(2)
            r3 = reward(3)
            writer.writerow([r1] + [r2] + [r3])

