import numpy as np
import pandas as pd

class Loader:
    def __init__(self, path):
        self.data=None
        self.table=None
        self.arm_num=None
        self.means=None
        self.optArm=None
        self.sampler=None
        self.range=None
        self.path=path
        self.load_data(path)

    def load_data(self,path):
        self.data=pd.read_csv(path)
        if 'arm1' in self.data.columns.values:
            self.load_data_gen()
        else:
            self.load_data_movie()

    def load_data_gen(self):
        path=self.path
        self.data = pd.read_csv(path)
        self.arm_num = self.data.columns.size
        self.range=self.data[self.data.columns[0]].max()-self.data[self.data.columns[0]].min()
        self.means = [0] * self.arm_num
        for i in range(self.arm_num):
            self.means[i] = self.data[self.data.columns[i]].mean()
        self.optArm = np.argmax(self.means)

        # 0,1,2,...
        self.table = np.zeros([self.arm_num, self.data.values.max() + 1, self.arm_num])
        for s_idx, source_arm in enumerate(self.data.columns):
            for a_idx, aim_arm in enumerate(self.data.columns):
                for i in range(self.data.values.max() + 1):
                    if source_arm == aim_arm:
                        self.table[s_idx][i][a_idx] = i
                    else:
                        p_reward = self.data[aim_arm][self.data[source_arm] == i].mean()
                        if p_reward > 1:
                            self.table[s_idx][i][a_idx] = 1
                        else:
                            self.table[s_idx][i][a_idx] = p_reward
        self.sampler=self.sample_gen

    def load_data_movie(self):
        #csv data
        self.arm_num=self.data['genre_col'].max()+1
        self.range=self.data['Rating'].max()-self.data['Rating'].min()
        self.means=[0]*self.arm_num
        for i in range(self.arm_num):
            self.means[i]=self.data[self.data['genre_col']==i]['Rating'].mean()
        self.optArm=np.argmax(self.means)
        #0,1,2,...
        self.table = np.zeros([self.arm_num, self.data['Rating'].max()-self.data['Rating'].min()+1, self.arm_num])
        for source in range(self.arm_num):
            for score in range(self.data['Rating'].min(),self.data['Rating'].max()+1):
                users=set(self.data[(self.data['genre_col'] == source) & (self.data['Rating'] == score)]['UserID'])
                self.table[source][:,source]=np.arange(self.data['Rating'].min(), self.data['Rating'].max() + 1)
                for aim in range(self.arm_num):
                    if aim==source:
                        continue
                    temp=self.data[self.data['genre_col'] == aim & self.data['UserID'].isin(users)]['Rating']
                    self.table[source][score-self.data['Rating'].min()][aim]=temp.mean()
        self.sampler =self.sample_movie

    def sample_movie(self,choose):
        v= self.data[self.data['genre_col'] == choose]['Rating'].sample(n=1, replace=True)
        return v.values[0]

    def sample_gen(self, choose):
        reward = self.data[self.data.columns[choose]].sample(n=1, replace=True)
        return reward.values[0]