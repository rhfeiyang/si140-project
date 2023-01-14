import numpy as np
import pandas as pd

class Loader:
    def __init__(self, path):
        self.data=None
        self.table=None
        self.arm_num=None
        self.means=None
        self.optArm=None
        self.load_data(path)
    def load_data(self,path):
        #csv data
        self.data = pd.read_csv(path)
        self.arm_num=self.data['genre_col'].max()+1
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
    def sample(self,choose):
        v= self.data[self.data['genre_col'] == choose]['Rating'].sample(n=1, replace=True)
        return v.values[0]