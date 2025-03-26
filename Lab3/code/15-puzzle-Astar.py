import heapq
import time

success = False
tar = {}
dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
st=set()
fa = {}

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

def get(a, x):
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
    start = tuple(tuple(row) for row in a)
    start_f = 0 + h(start)
    q = []
    heapq.heappush(q, (start_f, 0, start))
    fa[start] = None
    global success
    success = False
    start_time = time.time()
    while q and not success:
        cur_f, u_g, u_a = heapq.heappop(q)
        if u_a in st:
            continue
        st.add(u_a)
        if h(u_a) == 0:
            success = True
            print(u_g)
            path = []
            def backtrace(state):
                if state == start:
                    pos = get(state, 0)
                    path.append(f"({pos[0]},{pos[1]})")
                    return
                backtrace(fa[state])
                pos = get(state, 0)
                path.append(f"({pos[0]},{pos[1]})")
            backtrace(u_a)
            print("->".join(path) + "->")
            break
        cur_a = [list(row) for row in u_a]
        x, y = get(cur_a, 0)
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 1 <= nx <= 4 and 1 <= ny <= 4:
                new_a = [row[:] for row in cur_a]
                new_a[x][y], new_a[nx][ny] = new_a[nx][ny], new_a[x][y]
                new_a = tuple(tuple(row) for row in new_a)
                new_g = u_g + 1
                if new_a not in st :
                    fa[new_a] = u_a
                    new_f = new_g + h(new_a)
                    heapq.heappush(q, (new_f, new_g, new_a))
    end_time = time.time()
    print(f"\nRunning time: {end_time - start_time}s")

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