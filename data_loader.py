import numpy as np
import pandas as pd

class Loader:
    def __init__(self, path_train, path_test):
        self.train_data=None
        self.test_data=None
        self.table=None
        self.arm_num = None
        self.means=None
        self.optArm=None
        self.sampler=None
        self.range=None
        self.r_mid=None
        self.path_train=path_train
        self.path_test=path_test
        self.load_data(path_train,path_test)

    def load_data(self,path_train, path_test):
        self.train_data=pd.read_csv(path_train, low_memory=False)
        self.test_data=pd.read_csv(path_test, low_memory=False)
        if 'arm1' in self.train_data.columns.values:
            self.load_data_gen()
        else:
            self.load_data_movie()

    def load_data_gen(self):
        self.arm_num = self.train_data.columns.size
        self.range= self.train_data[self.train_data.columns[0]].max() - self.train_data[self.train_data.columns[0]].min()
        self.r_mid= float(self.train_data[self.train_data.columns[0]].max() + self.train_data[self.train_data.columns[0]].min()) / 2
        self.means = [0] * self.arm_num
        for i in range(self.arm_num):
            self.means[i] = self.test_data[self.test_data.columns[i]].mean()
        self.optArm = np.argmax(self.means)

        # 0,1,2,...
        self.table = np.zeros([self.arm_num, self.train_data.values.max() + 1, self.arm_num])
        for s_idx, source_arm in enumerate(self.train_data.columns):
            for a_idx, aim_arm in enumerate(self.train_data.columns):
                for i in range(self.train_data.values.max() + 1):
                    if source_arm == aim_arm:
                        self.table[s_idx][i][a_idx] = i
                    else:
                        p_reward = self.train_data[aim_arm][self.train_data[source_arm] == i].mean()
                        if p_reward > 1:
                            self.table[s_idx][i][a_idx] = 1
                        else:
                            self.table[s_idx][i][a_idx] = p_reward
        self.sampler=self.sample_gen

    def load_data_movie(self):
        #csv train_data
        self.arm_num = self.train_data['genre_col'].max() + 1
        self.range= self.train_data['Rating'].max() - self.train_data['Rating'].min()
        self.r_mid= float(self.train_data[self.train_data.columns[0]].max() + self.train_data[self.train_data.columns[0]].min()) / 2
        self.means=[0]*self.arm_num
        for i in range(self.arm_num):
            self.means[i]=self.test_data[self.test_data['genre_col'] == i]['Rating'].mean()
        self.optArm=np.argmax(self.means)
        #0,1,2,...
        self.table = np.zeros([self.arm_num, self.range + 1, self.arm_num])
        for source in range(self.arm_num):
            for score in range(self.train_data['Rating'].min(), self.train_data['Rating'].max() + 1):
                users=set(self.train_data[(self.train_data['genre_col'] == source) & (self.train_data['Rating'] == score)]['UserID'])
                self.table[source][:,source]=np.arange(self.train_data['Rating'].min(), self.train_data['Rating'].max() + 1)
                for aim in range(self.arm_num):
                    if aim==source:
                        continue
                    temp=self.train_data[(self.train_data['genre_col'] == aim) & (self.train_data['UserID'].isin(users))]['Rating']
                    self.table[source][score - self.train_data['Rating'].min()][aim]=temp.mean()
        self.sampler =self.sample_movie

    def sample_movie(self,choose):
        v= self.test_data[self.test_data['genre_col'] == choose]['Rating'].sample(n=1, replace=True)
        return v.values[0]

    def sample_gen(self, choose):
        reward = self.test_data[self.test_data.columns[choose]].sample(n=1, replace=True)
        return reward.values[0]