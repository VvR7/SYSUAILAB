#include<bits/stdc++.h>
using namespace std;
typedef pair<int,int>pii;
int success=0;
pii get(const vector<vector<int>>&a,int x)
{
    for (int i=1;i<=4;i++)
    {
        for (int j=1;j<=4;j++)
          if (a[i][j]==x) return {i,j};
    }
    return {0,0};
}
map<int,pii>tar;
void init()
{
    tar[1]={1,1};tar[2]={1,2};tar[3]={1,3};tar[4]={1,4};
    tar[5]={2,1};tar[6]={2,2};tar[7]={2,3};tar[8]={2,4};
    tar[9]={3,1};tar[10]={3,2};tar[11]={3,3};tar[12]={3,4};
    tar[13]={4,1};tar[14]={4,2};tar[15]={4,3};tar[16]={4,4};
}
int h(const vector<vector<int>>&a)
{
    int ans=0;
    for (int i=1;i<=4;i++)
    {
        for (int j=1;j<=4;j++)
          {
             if (a[i][j]==0) continue;
             pii x=tar[a[i][j]];
             ans+=abs(x.first-i)+abs(x.second-j);
          }
    }
    return ans;
}
int calc(const vector<vector<int>>&a)
{
    vector<int>d;
    int row=0;
    for (int i=1;i<=4;i++)
      for (int j=1;j<=4;j++)
        if (a[i][j]!=0) d.push_back(a[i][j]); else row=5-i;
    int ans=0;
    for (int i=0;i<15;i++)
      for (int j=0;j<i;j++)
        ans+=(d[j]>d[i]);
    return ans+row;
}
struct node{
    vector<vector<int>>a;
    int g=0;
    node(vector<vector<int>>a,int g):a(a),g(g) {}
    bool operator<(const node& other) const {
        return g+h(a)<other.g+h(other.a);
    };
    bool operator>(const node& other) const {
        return g+h(a)>other.g+h(other.a);
    };
};
int dx[4]={0,0,1,-1},dy[4]={1,-1,0,0};
int main()
{
    init();
    vector<vector<int>>a(5,vector<int>(5,0));
    int sx,sy;
    for (int i=1;i<=4;i++)
      for (int j=1;j<=4;j++)
        {
            cin>>a[i][j];
            if (a[i][j]==0) {
                sx=i;sy=j;
            }
        }
    clock_t begin=clock();
    if (calc(a)%2==0) {
        cout<<"Can't Solve"<<"\n";
        return 0;
    }
    vector<pii>ans;
    ans.push_back({sx,sy});
    function<void(vector<vector<int>>&,map<vector<vector<int>>,vector<vector<int>>>&)>backtrace=[&](const vector<vector<int>>&x,map<vector<vector<int>>,vector<vector<int>>>&fa) {
        if (x==a) {
            pii u=get(x,0);
            cout<<"("<<u.first<<","<<u.second<<")->";
            return;
        }
        backtrace(fa[x],fa);
        pii u=get(x,0);
        cout<<"("<<u.first<<","<<u.second<<")->";
    };
    function<void(int)>dfs=[&](int maxdep){
        priority_queue<node,vector<node>,greater<node>>q;
        map<vector<vector<int>>,int>mp;
        map<vector<vector<int>>,vector<vector<int>>>fa;
        node start=node(a,0);
        mp[a]=0;
        q.push(start);
        while(!success&&q.size()) {
            node u=q.top();q.pop();
            if (h(u.a)==0) {
                success=1;
                cout<<u.g<<"\n";
                backtrace(u.a,fa);
                break;
            }
            if (mp.count(u.a)&&u.g>mp[u.a]) continue;
            if (u.g+h(u.a)>maxdep) break;
            int g=u.g;
            int x,y;
            x=get(u.a,0).first; y=get(u.a,0).second;
            for (int i=0;i<4;i++)
            {
                int nx=x+dx[i],ny=y+dy[i];
                if (nx<1||ny<1||nx>4||ny>4) continue;
                vector<vector<int>>na=u.a;
                swap(na[x][y],na[nx][ny]);
                if (!mp.count(na)||g+1<mp[na])
                {
                    fa[na]=u.a;
                    mp[na]=g+1;
                    q.push(node(na,g+1));
                }
            }
        }
    };
    for (int maxdep=0;success==0;maxdep++)
    {
        dfs(maxdep);
        clock_t end=clock();
        cout<<"step:"<<maxdep<<"  "<<"Running time:"<<double(end-begin)/CLOCKS_PER_SEC<<"s"<<endl;
    }
    clock_t end=clock();
    cout<<"\nRunning time:"<<double(end-begin)/CLOCKS_PER_SEC<<"s"<<endl;
}