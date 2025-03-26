import time
import heapq
import sys

success = 0
tar = {}

def init():
    global tar
    tar[1]  = (1, 1);  tar[2]  = (1, 2);  tar[3]  = (1, 3);  tar[4]  = (1, 4)
    tar[5]  = (2, 1);  tar[6]  = (2, 2);  tar[7]  = (2, 3);  tar[8]  = (2, 4)
    tar[9]  = (3, 1);  tar[10] = (3, 2);  tar[11] = (3, 3);  tar[12] = (3, 4)
    tar[13] = (4, 1);  tar[14] = (4, 2);  tar[15] = (4, 3);  tar[16] = (4, 4)

def get(a, x):
    # 遍历 1~4 的行和列（下标 0 保留未使用）
    for i in range(1, 5):
        for j in range(1, 5):
            if a[i][j] == x:
                return (i, j)
    return (0, 0)

def h(a):
    ans = 0
    for i in range(1, 5):
        for j in range(1, 5):
            if a[i][j] == 0:
                continue
            tx,ty = tar.get(a[i][j], (0, 0))
            ans += abs(tx - i) + abs(ty - j)

            if i==tx:
                for jj in range(j+1,5):
                    tx2,ty2=tar.get(a[i][jj],(0,0))
                    if i==tx2:
                        if (j-jj)*(ty-ty2)<0:
                            ans+=2

            if j==ty:
                for ii in range(i+1,5):
                    tx2,ty2=tar.get(a[ii][j],(0,0))
                    if j==ty2:
                        if (i-ii)*(tx-tx2)<0:
                            ans+=2


    return ans

def calc(a):
    d = []
    row = 0
    for i in range(1, 5):
        for j in range(1, 5):
            if a[i][j] != 0:
                d.append(a[i][j])
            else:
                row = 5 - i
    ans = 0
    for i in range(15):
        for j in range(i):
            if d[j] > d[i]:
                ans += 1
    return ans + row

class node:
    def __init__(self, a, g):
        self.a = a  # 棋盘状态（列表的列表，保持 5x5，0 号行未使用）
        self.g = g
    def __lt__(self, other):
        return self.g + h(self.a) < other.g + h(other.a)

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

def board_to_key(a):
    # 将棋盘状态转换为 tuple-of-tuples 以便作为 dict 的 key
    return tuple(tuple(row) for row in a)

def main():
    global success
    init()
    # 初始化 5x5 棋盘，下标 0 保留未使用
    a = [[0] * 5 for _ in range(5)]
    sx = sy = 0
    # 读入 4x4 的棋盘数据（从下标 1 到 4）
    for i in range(1, 5):
        # 每行输入 4 个数字，以空格分隔
        line = sys.stdin.readline().strip().split()
        if not line:
            break
        for j in range(1, 5):
            a[i][j] = int(line[j-1])
            if a[i][j] == 0:
                sx, sy = i, j

    begin = time.time()
    # 如果不可解，输出提示并退出
    if calc(a) % 2 == 0:
        print("Can't Solve")
        return

    # 保存初始状态的 key
    initial_board_key = board_to_key(a)

    def backtrace(x, fa):
        # 当到达初始状态时输出空格位置
        if x == initial_board_key:
            u = get(x, 0)
            print("(%d,%d)->" % (u[0], u[1]), end="")
            return
        backtrace(fa[x], fa)
        u = get(x, 0)
        print("(%d,%d)->" % (u[0], u[1]), end="")

    def dfs(maxdep):
        global success
        q = []
        mp = {}  # 记录每个状态的最小 g 值
        fa = {}  # 状态转移的父状态，用于回溯路径
        start = node(a, 0)
        key_a = board_to_key(a)
        mp[key_a] = 0
        heapq.heappush(q, start)
        while not success and q:
            u = heapq.heappop(q)
            key_u = board_to_key(u.a)
            if h(u.a) == 0:
                success = 1
                print(u.g)
                backtrace(key_u, fa)
                break
            if key_u in mp and u.g > mp[key_u]:
                continue
            if u.g + h(u.a) > maxdep:
                # 当前状态不在 maxdep 限制内
                continue
            x, y = get(u.a, 0)
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]
                if nx < 1 or ny < 1 or nx > 4 or ny > 4:
                    continue
                # 复制当前状态，交换空格与相邻数字
                na = [row[:] for row in u.a]
                na[x][y], na[nx][ny] = na[nx][ny], na[x][y]
                key_na = board_to_key(na)
                if key_na not in mp or u.g + 1 < mp[key_na]:
                    fa[key_na] = key_u
                    mp[key_na] = u.g + 1
                    heapq.heappush(q, node(na, u.g + 1))
    
    maxdep = 0
    while success == 0:
        dfs(maxdep)
        end = time.time()
        print("step:%d  Running time:%fs" % (maxdep, end - begin))
        maxdep += 1

    end = time.time()
    print("\nRunning time:%fs" % (end - begin))

if __name__ == "__main__":
    main()
