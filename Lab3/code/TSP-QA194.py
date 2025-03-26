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
            self.all_history_FitV.append(self.fit_value.copy())
            self.mutation_prob=self.mutation_prob*self.gamma

            # if i%100==99:
            #     best_index=np.array(self.generation_best_Y).argmin()
            #     best_x=self.generation_best_X[best_index]
            #     best_y = self.F(np.array([best_x]))[0]
            #     print(f'iteration:{i} best_y:{best_y}')
            best_index=np.array(self.generation_best_Y).argmin()
            best_x=self.generation_best_X[best_index]
            best_y = self.F(np.array([best_x]))[0]
            self.all_history_Y.append(best_y)
            print(f'iteration:{i} best_y:{best_y}')
        global_best_index=np.array(self.generation_best_Y).argmin()
        self.best_x=self.generation_best_X[global_best_index]
        self.best_y = self.F(np.array([self.best_x]))[0]
        return self.best_x,self.best_y

def read_tsp_file(file_path):     
    coordinates = []     
    try:         
        with open(file_path, 'r') as file:             
            for line in file:                 
                parts = line.strip().split()                 
                if len(parts) == 3:                     
                    x = float(parts[1])                     
                    y = float(parts[2])                     
                    coordinates.append((x, y))     
    except FileNotFoundError:         
        print(f"错误: 文件 {file_path} 未找到。")     
    except Exception as e:         
        print(f"错误: 发生未知错误: {e}")     
    return coordinates

if __name__ == "__main__":
    
    file_path = 'QA194.txt'
    points_coordinate = read_tsp_file(file_path)
    num_points=len(points_coordinate)
    print(num_points)
    from scipy.spatial import distance


    def get_distance(i, j):
        return distance.euclidean(points_coordinate[i], points_coordinate[j])


    # 在 cal_total_distance() 中改用 get_distance()
    def cal_total_distance(routine):
        size_pop, n_points = routine.shape
        output = []
        for i in range(size_pop):
            total_dist = sum(get_distance(routine[i, j], routine[i, (j + 1) % n_points]) for j in range(n_points))
            output.append(total_dist)
        return np.array(output)


    ga_tsp=GA_TSP(F=cal_total_distance,n_dim=num_points,population_size=500,max_iter=1000,mutation_prob=0.2,gamma=1)

    best_points,best_distance=ga_tsp.run()

    print(best_points,best_distance)

    '''最优解9352'''


    '''可视化'''
    import matplotlib.pyplot as plt

    plt.plot(range(ga_tsp.max_iter), ga_tsp.all_history_Y, linestyle='-', color='b', label="Fitness Trend")
    plt.title("Genetic Algorithm TSP Optimization Progress")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Value")
    plt.legend()
    plt.show()