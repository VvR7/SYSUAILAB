import heapq
import time

success = False
tar = {}
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
st = set()
fa = {}

shift_map = [
    [0,  0,  0,  0,   0],
    [0,  60, 56, 52, 48],
    [0,  44, 40, 36, 32],
    [0,  28, 24, 20, 16],
    [0,  12, 8,   4,  0]
]

def init():
    global tar
    tar[1] = (1, 1)
    tar[2] = (1, 2)
    tar[3] = (1, 3)
    tar[4] = (1, 4)
    tar[5] = (2, 1)
    tar[6] = (2, 2)
    tar[7] = (2, 3)
    tar[8] = (2, 4)
    tar[9] = (3, 1)
    tar[10] = (3, 2)
    tar[11] = (3, 3)
    tar[12] = (3, 4)
    tar[13] = (4, 1)
    tar[14] = (4, 2)
    tar[15] = (4, 3)

def array_to_int(a):
    num = 0
    for i in range(1, 5):
        for j in range(1, 5):
            num = (num << 4) | a[i][j]
    return num

def h(num):
    ans = 0
    for i in range(1,5):
        for j in range(1,5):
            shift = shift_map[i][j]
            val = (num >> shift) & 0xF
            if val == 0:
                continue
            tx, ty = tar[val]
            ans += abs(i - tx) + abs(j - ty)
            # Check row conflicts
            if i == tx:
                for jj in range(j + 1, 5):
                    shift_jj = shift_map[i][jj]
                    val_jj = (num >> shift_jj) & 0xF
                    if val_jj==0:
                        continue
                    tx_jj, ty_jj = tar[val_jj]
                    if tx_jj == i:
                        if (ty < ty_jj) != (j < jj):
                            ans += 2
            if j == ty:
                for ii in range(i + 1, 5):
                    shift_ii = shift_map[ii][j]
                    val_ii = (num >> shift_ii) & 0xF
                    if val_ii==0:
                        continue
                    tx_ii, ty_ii = tar[val_ii]
                    if ty_ii == j:
                        if (tx < tx_ii) != (i < ii):
                            ans += 2
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
    for i in range(len(d)):
        for j in range(i):
            if d[j] > d[i]:
                ans += 1
    return ans + row

def main():
    init()
    a = [[0] * 5 for _ in range(5)]
    sx, sy = 0, 0
    for i in range(1, 5):
        row = list(map(int, input().split()))
        for j in range(1, 5):
            a[i][j] = row[j-1]
            if a[i][j] == 0:
                sx, sy = i, j
    if calc(a) % 2 == 0:
        print("Can't Solve")
        return
    start = array_to_int(a)
    start_f = 0 + h(start)
    q = []
    heapq.heappush(q, (start_f, 0, start, sx, sy,-1))
    fa[start] = -1
    global success
    success = False
    start_time = time.time()
    node_count = 0
    while q and not success:
        cur_f, u_g, u_s, u_x, u_y,fa_s = heapq.heappop(q)
        if u_s in st:
            continue
        fa[u_s]=fa_s
        node_count += 1
        st.add(u_s)
        if h(u_s) == 0:
            success = True
            print(u_g)
            path = []
            def backtrace(state):
                if fa.get(state) == -1:
                    x, y = int_to_pos(state)
                    path.append(f"({x},{y})")
                    return
                backtrace(fa[state])
                x, y = int_to_pos(state)
                path.append(f"({x},{y})")
            backtrace(u_s)
            print("->".join(path))
            break
        x, y = u_x, u_y
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 1 <= nx <= 4 and 1 <= ny <= 4:
                shift1 = shift_map[x][y] 
                shift2 = shift_map[nx][ny]
                val2 = (u_s >> shift2) & 0xF  #要交换的值
                mask = (0xF << shift1) | (0xF << shift2)  #掩码:把val2原本的位置变为空格
                new_s = u_s & ~mask  
                new_s |= (val2 << shift1)  #把val2放在原本空格的位置
                new_g = u_g + 1
                if new_s not in st and new_s not in fa:
                    new_f = new_g + h(new_s)
                    heapq.heappush(q, (new_f, new_g, new_s, nx, ny,u_s))
        if node_count % 1000000 == 0:
            print(f"Nodes processed: {node_count}, Time elapsed: {time.time() - start_time}s")
    end_time = time.time()
    print(f"Total running time: {end_time - start_time}s")
    print(f'total explore nodes: {node_count}')
def int_to_pos(num):
    for i in range(1,5):
        for j in range(1,5):
            shift = shift_map[i][j]
            val = (num >> shift) & 0xF
            if val == 0:
                return (i, j)
    return (0, 0)

if __name__ == "__main__":
    main()

'''
样例一
1 2 4 8
5 7 11 10
13 15 0 3
14 6 9 12
answer:22

样例二
14 10 6 0
4 9 1 8
2 3 5 11
12 13 7 15

answer:49

样例三
5 1 3 4
2 7 8 12
9 6 11 15
0 13 10 14

answer:15

样例四
6 10 3 15
14 8 7 11
5 1 0 2
13 12 9 4

answer:48

样例五
11 3 1 7
4 6 8 2
15 9 10 13
14 12 5 0

answer:56

样例六
0 5 15 14
7 9 6 13
1 2 12 10
8 11 4 3

answer:62
'''