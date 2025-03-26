'''判断是否是变量'''
def isvariable(x):
    vars={'u','uu','v','vv','w','ww','x','xx','y','yy','z','zz'}
    return x in vars
'''判断是否是复合项'''
def isfunc(x):
    return isinstance(x,str) and '(' in x
'''把复合项拆开为 谓词名+参数列表'''
def Split(x):
    args=[]
    start=x.find('(')
    func_name=x[:start]
    args_str=x[start+1:-1]
    s=''
    dep=0
    for c in args_str:
        if c=='(':
            dep+=1
        elif c==')':
            dep-=1
        if c==',' and dep==0:
            args.append(s)
            s=''
        else:
            s+=c
    if s:
        args.append(s.strip())
    return func_name,args
'''把term应用替换映射'''
def Map(term,dictionary):
    if not dictionary:
        return term
    if isvariable(term):
        if term in dictionary:
            return Map(dictionary[term],term)
        else:
            return term
    elif isfunc(term):
        func_name,args=Split(term)
        new_args=[Map(arg,dictionary) for arg in args]
        return f'{func_name}({','.join(new_args)})'
    else:
        return term
def occurs_check(v,term):#检查变量v是否在term中出现
    if v==term:
        return True
    if isfunc(term):
        _,args=Split(term)
        for t in args:
            if occurs_check(v,t):
                return True
    return False
'''对两个原子公式的列表进行合一'''
def MGU(para1,para2):
    if len(para1) != len(para2):
        return None
    subst={}
    equa=list(zip(para1,para2))
    while equa:
        s,t=equa.pop(0)
        s=Map(s,subst)
        t=Map(t,subst)
        if s==t:
            continue
        if isvariable(s):
            if occurs_check(s,t):
                return None
            subst[s]=t
            continue
        if isvariable(t):
            if occurs_check(t,s):
                return None
            subst[t]=s
            continue
        if isfunc(s) and isfunc(t):
            s_name,s_args=Split(s)
            t_name,t_args=Split(t)
            if s_name != t_name or len(s_args)!=len(t_args):
                return None
            equa.extend(zip(s_args,t_args))
            continue
        return None
    for x in list(subst.keys()):
        subst[x]=Map(subst[x],subst)
    return subst
if __name__=='__main__':
    para1 = ['x']
    para2 = ['f(x,y)']
    result1 = MGU(para1, para2)
    para1 = ['P(a,x,f(g(y)))']
    para2 = ['P(z,f(z),f(u))']
    result2 = MGU(para1, para2)
    para1 = ['F(x, g(y, x))']
    para2 = ['F(h(z), g(a, h(z)))']
    result3 = MGU(para1,para2)
    para1 = ['P(f(x), x)']
    para2 = ['P(f(a), b)']
    result4 = MGU(para1,para2)
    para1 = ['P(xx,a)']
    para2 = ['P(b,yy)']
    result5 = MGU(para1,para2)
    para1 = ['P(a,xx,f(g(yy)))']
    para2 = ['P(zz,f(zz),f(uu))']
    result6 = MGU(para1,para2)
    print(result3)