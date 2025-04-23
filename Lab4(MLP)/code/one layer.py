import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

df=pd.read_excel('MLP_data.xlsx')
x=df[['housing_age','homeowner_income']].values
y=df['house_price'].values
MAX_x=np.max(x,axis=0)
MIN_x=np.min(x,axis=0)
MAX_y=np.max(y)
MIN_y=np.min(y)
def transform_data(x,x_min,x_max):
    return (x-x_min)/(x_max-x_min)
def inverse_transform(x,x_min,x_max):
    return x*(x_max-x_min)+x_min
x=transform_data(x,MIN_x,MAX_x)
y=transform_data(y,MIN_y,MAX_y)
def preprocess_data(x,y):
    idx=list(range(len(x)))
    random.shuffle(idx)
    train_idx=idx[:int(0.7*len(idx))]
    val_idx=idx[int(0.7*len(idx)):int(0.85*len(idx))]
    test_idx=idx[int(0.85*len(idx)):]

    x_train=x[train_idx]
    y_train=y[train_idx]
    x_val=x[val_idx]
    y_val=y[val_idx]
    x_test=x[test_idx]
    y_test=y[test_idx]

    return x_train,y_train,x_val,y_val,x_test,y_test

x_train,y_train,x_val,y_val,x_test,y_test=preprocess_data(x,y)
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr

        # He 初始化
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.random.randn(hidden_dim) * np.sqrt(2 / input_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.random.randn(output_dim) * np.sqrt(2 / hidden_dim)
        self.train_loss_list = []
        self.val_loss_list = []

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # 缓存输入，用于反向传播
        self.x = x

        # 隐藏层
        self.z1 = x.dot(self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        # 输出层（线性）
        self.z2 = self.a1.dot(self.w2) + self.b2
        return self.z2

    def loss(self, y_pred, y_true):
        # 均方误差
        y_pred=y_pred.reshape(-1)
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_true):
        # 假设 y_true 形状为 (batch_size,) 或 (batch_size,1)
        y = y_true.reshape(-1, 1)
        batch_size = y.shape[0]

        # 输出层梯度
        grad_z2 = 2 * (self.z2 - y) / batch_size
        grad_w2 = self.a1.T.dot(grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)

        # 隐藏层梯度
        grad_a1 = grad_z2.dot(self.w2.T)
        grad_z1 = grad_a1 * (self.z1 > 0)
        grad_w1 = self.x.T.dot(grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)

        # 参数更新
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

    def save_model(self, path):
        np.savez(
            path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            train_loss_list=self.train_loss_list,
            val_loss_list=self.val_loss_list
        )

    def load_model(self, path):
        params = np.load(path, allow_pickle=True)
        self.w1 = params['w1']
        self.b1 = params['b1']
        self.w2 = params['w2']
        self.b2 = params['b2']
        self.train_loss_list = params['train_loss_list'].tolist()
        self.val_loss_list = params['val_loss_list'].tolist()

    def train(self,x_train,y_train,x_val,y_val,epochs,lr,batch_size=32,discount_rate=0.9998,constrant=1e-7):
        self.lr=lr
        min_lr=1e-7
        val_best_loss=float('inf')
        n_samples=x_train.shape[0]
        stop_count=0
        for epoch in range(epochs):
            idx=np.random.permutation(n_samples)
            x_train=x_train[idx]
            y_train=y_train[idx]
            train_loss=0
            
            self.lr=max(self.lr*discount_rate,min_lr)
            
            for i in range(0,n_samples,batch_size):
                x_batch=x_train[i:i+batch_size]
                y_batch=y_train[i:i+batch_size]
                pred=self.forward(x_batch)
                self.loss(pred,y_batch)
                self.backward(y_batch)
            train_pred=self.forward(x_train)
            train_loss=self.loss(train_pred,y_train)
            self.train_loss_list.append(train_loss)

            #validation
            val_pred=self.forward(x_val)
            val_loss=self.loss(val_pred,y_val)
            self.val_loss_list.append(val_loss)
            if val_loss<val_best_loss :
                if val_best_loss-val_loss<constrant:
                    stop_count+=1
                    if stop_count>=200:
                        break
                else:
                    stop_count=0
                val_best_loss=val_loss
                self.save_model('best_model')
                
            else:
                stop_count+=1
                if stop_count>=100:
                    break
            if epoch%10==0:
                print(f'Epoch {epoch}, Train Loss: {train_loss}, Val best Loss: {val_best_loss}, LR: {self.lr:.2e}')
    def test(self,x_test,y_test):
        pred=self.forward(x_test)
        test_loss=self.loss(pred,y_test)
        return test_loss,pred

model=MLP(input_dim=2,hidden_dim=8,output_dim=1)
model.train(x_train,y_train,x_val,y_val,epochs=50000,lr=0.05,batch_size=len(x_train),discount_rate=0.9999)
model.load_model('best_model.npz')
test_loss,pred=model.test(x_test,y_test)
print(f'Test Loss: {test_loss}')

# print(model.train_loss_list)
# print(model.val_loss_list)
# print(f'Predictions vs True:')
# for i in range(len(pred)):
#     pred_inv=inverse_transform(pred[i].item(),MIN_y,MAX_y)
#     true_inv=inverse_transform(y_test[i].item(),MIN_y,MAX_y)
#     print(f'Pred: {pred_inv:.3f}, True: {true_inv:.3f}')



#================================================================================
#可视化
train_loss_list=model.train_loss_list
val_loss_list=model.val_loss_list

plt.figure(figsize=(12, 5))  # 设置画布大小

# 第一个子图：训练损失
plt.subplot(1, 2, 1)  # 1行2列的第1个子图
plt.plot(train_loss_list, label='Train Loss', color='blue')
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 第二个子图：验证损失
plt.subplot(1, 2, 2)  # 1行2列的第2个子图
plt.plot(val_loss_list, label='Validation Loss', color='orange')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()  # 自动调整子图间距，避免重叠
plt.show()

pred=[i[0] for i in pred]
pred=[inverse_transform(i,MIN_y,MAX_y) for i in pred]
y_test=[inverse_transform(i,MIN_y,MAX_y) for i in y_test]
plt.plot([0,600000],[0,600000],'r-',label='pred=True') # 添加y=x直线
plt.plot(pred,y_test,'.')
plt.xlim(0,600000)
plt.ylim(0,600000)
plt.xlabel('Pred')
plt.ylabel('True')
plt.show()




n_samples = 500

col1 = np.random.uniform(0, 60, n_samples)
col1=[transform_data(i,MIN_x[0],MAX_x[0]) for i in col1]
col2 = np.random.uniform(0, 20, n_samples)
col2=[transform_data(i,MIN_x[1],MAX_x[1]) for i in col2]

X = np.column_stack((col1, col2))
y=model.forward(X).reshape(-1)

y=[inverse_transform(i,MIN_y,MAX_y) for i in y]

col1=[inverse_transform(i,MIN_x[0],MAX_x[0]) for i in col1]
col2=[inverse_transform(i,MIN_x[1],MAX_x[1]) for i in col2]
X=np.column_stack((col1,col2))


y=[i/10000 for i in y]
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
coef_age, coef_income = model.coef_
intercept = model.intercept_

# 创建平面网格
age_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
income_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
age_grid, income_grid = np.meshgrid(age_vals, income_vals)

# 计算平面上的预测房价
price_grid = intercept + coef_age * age_grid + coef_income * income_grid

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 渐变色散点图
scatter = ax.scatter(
    X[:, 0], X[:, 1], y,
    c=y,
    cmap='viridis',
    marker='o',
    s=20,
    alpha=0.8
)

# 拟合平面（半透明）
ax.plot_surface(
    age_grid, income_grid, price_grid,
    alpha=0.4,
    cmap='viridis',
    edgecolor='none'
)

# 标签与颜色条

ax.set_xlabel('Housing Age')
ax.set_ylabel('Homeowner Income')

fig.colorbar(scatter, ax=ax, label='House Price(*10k)')
plt.title('hidden dim:8')
plt.show()






# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 使用 y 值作为颜色映射，cmap 选择常用的 'viridis' 或 'plasma' 等
# scatter = ax.scatter(
#     X[:, 0],        # x: housing_age
#     X[:, 1],        # y: homeowner_income
#     y,              # z: house_price
#     c=y,            # 用 y 值控制颜色深浅
#     cmap='viridis', # 颜色映射风格，可换成 'plasma', 'coolwarm' 等
#     marker='o',
#     s=20,
#     alpha=0.8
# )

# ax.set_xlabel('Housing Age')
# ax.set_ylabel('Homeowner Income')
# ax.set_zlabel('House Price')

# # 添加颜色条，显示房价与颜色的关系
# fig.colorbar(scatter, ax=ax, label='House Price')

# plt.show()











        



