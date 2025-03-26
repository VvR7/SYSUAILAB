import numpy as np
from scipy import spatial
import sko.GA as GA
import pandas as pd
class GA_TSP:
    def __init__(self,F,n_dim,population_size,max_iter,mutation_prob,gamma=0.99):
        self.F=F   #函数
        self.n_dim=n_dim   #维度
        self.population_size=population_size
        self.max_iter=max_iter
        self.mutation_prob=mutation_prob
        self.X=None   #population_size*n_dim
        self.Y=None   #population_size,1
        self.fit_value=None #population_size,1
        self.gamma=gamma
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.best_x, self.best_y = None, None
        
        self.create()
    # x2y默认为self.F(self.X)
    def create(self):  #创造初始种群
        '''创建一个全排列'''
        x=np.random.rand(self.population_size,self.n_dim)
        self.X=x.argsort(axis=1)
    def rank(self): #计算适应度
        self.fit_value=-self.Y
    def mutation(self): #变异
        '''交换两个点：效果很差'''
        # for i in range(self.population_size):
        #     for j in range(self.n_dim):
        #         if np.random.rand()<self.mutation_prob:
        #             k=np.random.randint(0,self.n_dim,1)
        #             self.X[i,j],self.X[i,k]=self.X[i,k],self.X[i,j]
        '''反转子段：效果很好'''
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_prob:
                j, k = np.sort(np.random.choice(self.n_dim, 2, replace=False))
                self.X[i, j:k+1] = self.X[i, j:k+1][::-1]
    def select(self,siz=3): #选择
        '''采用锦标赛方式:每次随机挑选siz个个体为候选者，从中选一个最优的，重复population_size次'''
        candidate_idx=np.random.randint(self.population_size,size=(self.population_size,siz))
        candidate_value=self.fit_value[candidate_idx]
        winner=candidate_value.argmax(axis=1)
        select_idx=[candidate_idx[i,j] for i,j in enumerate(winner)]
        self.X=self.X[select_idx,:]
    def cross(self,cross_prob=0.8): #交叉,生成后代
        '''排列交叉，维护基因唯一性'''
        for i in range(0,self.population_size,2):
            if np.random.rand()<cross_prob:
                p,k=np.random.randint(0,self.n_dim,2)
                if p>k:
                    p,k=k,p
                mp1={value: idx for idx,value in enumerate(self.X[i])}
                mp2={value: idx for idx,value in enumerate(self.X[i+1])}
                for j in range(p,k):
                    val1,val2=self.X[i,j],self.X[i+1,j]
                    pos1,pos2=mp1[val2],mp2[val1]
                    self.X[i,j],self.X[i,pos1]=self.X[i,pos1],self.X[i,j]
                    self.X[i+1,j],self.X[i+1,pos2]=self.X[i+1,pos2],self.X[i+1,j]
                    mp1[val1],mp1[val2]=pos1,j
                    mp2[val1],mp2[val2]=j,pos2
    def run(self):
        for i in range(self.max_iter):
            OLD_X=self.X.copy()
            self.Y=self.F(self.X)
            self.rank()
            self.select()
            self.cross()
            self.mutation()

            self.X=np.concatenate([OLD_X,self.X],axis=0)
            self.Y=self.F(self.X)
            self.rank()
            select_idx=np.argsort(self.Y)[:self.population_size]
            self.X=self.X[select_idx,:]
            self.Y = self.Y[select_idx]
            self.rank()

            '''记录最优'''
            best_index=self.fit_value.argmax()
            self.generation_best_X.append(self.X[best_index,:].copy())
            self.generation_best_Y.append(self.Y[best_index])
            self.all_history_Y.append(self.Y.copy())
            self.all_history_FitV.append(self.fit_value.copy())
            self.mutation_prob=self.mutation_prob*self.gamma
        global_best_index=np.array(self.generation_best_Y).argmin()
        self.best_x=self.generation_best_X[global_best_index]
        self.best_y = self.F(np.array([self.best_x]))[0]
        return self.best_x,self.best_y

if __name__=='__main__':
    num_points = 50
    # points_coordinate = np.random.rand(num_points, 2)
    # distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
    
    # df_distance = pd.DataFrame(distance_matrix)

    # # 写入 Excel 文件
    # df_distance.to_excel("distance_matrix.xlsx", index=False)
    # print("Distance matrix 已保存到 distance_matrix.xlsx")

    df_loaded = pd.read_excel("distance_matrix.xlsx")
    distance_matrix= df_loaded.values
    def cal_total_distance(routine):
        '''
        输入：routine为形状 (size_pop, num_points) 的排列矩阵
        输出：返回每个排列对应的路径总距离，数组形状为 (size_pop,)
        '''
        size_pop, n_points = routine.shape
        output = []
        for i in range(size_pop):
            total_dist = 0
            for j in range(n_points):
                total_dist += distance_matrix[routine[i, j], routine[i, (j + 1) % n_points]]
            output.append(total_dist)
        return np.array(output)
    
    ga_tsp=GA_TSP(F=cal_total_distance,n_dim=num_points,population_size=50,max_iter=500,mutation_prob=0.2,gamma=0.999)

    best_points,best_distance=ga_tsp.run()

    print(best_points,best_distance)



    