from MGU import *
def Index(literal_idx,clause_idx,len):
    if len==1:
        return str(clause_idx+1)
    else:
        return str(clause_idx+1)+chr(ord('a')+literal_idx)
def iscomplementary(x,y):
    if not x or not y:  
        return False
    endx=x.find('(')
    endy=y.find('(')
    if x[0]=='~' and x[1:endx]==y[:endy]:
        return True
    if y[0]=='~' and y[1:endy]==x[:endx]:
        return True
    return False
def resolve(clause1,clause2,idx1,idx2):   #归结得到新子句
    newclause=list(clause1)+list(clause2)
    newclause.remove(clause1[idx1])
    newclause.remove(clause2[idx2])
    newclause=list(dict.fromkeys(newclause))
    return tuple(newclause)
def sequence(newclause,idx1,idx2,dictionary):
    if not dictionary:
        ans='R['+idx1+','+idx2+']='
    else:
        ans='R['+idx1+','+idx2+']'
        for key,value in dictionary.items():
            ans+='{'+str(key)+'='+str(value)+'}'
        ans+='='
    ans+=str(newclause)
    return ans
def sub(clause,dictionary):
    newclause=[]
    for x in clause:
        newclause.append(Map(x,dictionary))
    return tuple(newclause)
def resolution(KB):
    ALL=list(KB)
    support_list=[ALL[-1]]
    result=[]
    vis=set()
    while True:
        newclauset=[]
        for clause1_idx in range(len(ALL)):
            for clause2_idx in range(clause1_idx+1,len(ALL)):
                if clause1_idx==clause2_idx :
                    continue
                clause1,clause2=ALL[clause1_idx],ALL[clause2_idx]
                if (clause1,clause2) in vis:
                    continue
                if clause2 not in support_list and clause1 not in support_list:
                    continue
                for literal_idx1 in range(len(clause1)):
                    for literal_idx2 in range(len(clause2)):
                        literal1,literal2=clause1[literal_idx1],clause2[literal_idx2]
                        if not iscomplementary(literal1,literal2):
                            continue
                        '''处理互补对'''
                        literal1=literal1.replace('~','')
                        literal2=literal2.replace('~','')
                        literal1,literal2=[literal1],[literal2]
                        mgu_dict=MGU(literal1,literal2)
                        if mgu_dict==None:
                            continue
                        mgu_clause1=sub(clause1,mgu_dict)
                        mgu_clause2=sub(clause2,mgu_dict)
                        newclause=resolve(mgu_clause1,mgu_clause2,literal_idx1,literal_idx2)
                        if newclause in ALL or newclause in newclauset:
                            continue
                        vis.add((clause1,clause2))
                        idx1=Index(literal_idx1,clause1_idx,len(clause1))
                        idx2=Index(literal_idx2,clause2_idx,len(clause2))
                        seq=sequence(newclause,idx1,idx2,mgu_dict)
                        result.append(seq)
                        newclauset.append(newclause)
                        if newclause==():
                            return result
        ALL.extend(newclauset)
        support_list.extend(newclauset)
def newnum(num,res,usefulres,size):
    if num<=size:
        return num
    fa_seq=res[num-1]
    start=fa_seq.find('(')
    for i in range(size,len(usefulres)):
        begin=usefulres[i].find('(')
        if usefulres[i][begin:]==fa_seq[start:]:
            return i+1

def getfa(clause):
    start=clause.find('[')
    end=clause.find(']')
    num=clause[start+1:end].split(',')
    fa1=int(''.join(x for x in num[0] if not x.isalpha()))
    fa2=int(''.join(x for x in num[1] if not x.isalpha()))
    return fa1,fa2

def Resequence(seq,num1,num2,newnum1,newnum2):
    num1_pos=seq.find(num1)
    end=num1_pos+len(num1)
    seq=seq[:num1_pos]+newnum1+seq[end:]
    findnum2=num1_pos+len(newnum1)
    num2_pos=seq.find(num2, findnum2)
    end=num2_pos+len(num2)
    seq=seq[:num2_pos]+newnum2+seq[end:]
    return seq
def simplify(res,size):   #size是初始子句集大小
    useful=[]
    que=[len(res)]
    vis=set()
    while que!=[]:
        front=que.pop(0)
        if front in vis:
            continue
        vis.add(front)

        useful.append(res[front-1])
        fa1,fa2=getfa(res[front-1])
        if fa1>size:
            que.append(fa1)
        if fa2>size:
            que.append(fa2)
    useful.reverse()
    usefulres=res[0:size]+useful
    for i in range(size,len(usefulres)):
        fa1,fa2=getfa(usefulres[i])
        newnum1=str(newnum(fa1,res,usefulres,size))
        newnum2=str(newnum(fa2,res,usefulres,size))
        #print(fa1,newnum1,fa2,newnum2)
        usefulres[i]=Resequence(usefulres[i],str(fa1),str(fa2),newnum1,newnum2)
    return usefulres
def solve(KB):
    res=list(KB.copy())+resolution(KB)
    res=simplify(res,len(KB))
    return res
if __name__=='__main__':
    KB1=[('GradStudent(sue)',),('~GradStudent(sue)','Student(x)'),('~Student(x)','Hardworker(x)'),('~Hardworker(sue)',)]
    KB2=[('A(tony)',),('A(mike)',),('A(john)',),('L(tony,rain)',),('L(tony,snow)',),('~A(x)','S(x)','C(x)'),('~C(y)','~L(y,rain)'),('L(z,snow)','~S(z)'),('~L(tony,u)','~L(mike,u)'),('L(tony,v)','L(mike,v)'),('~A(w)','~C(w)','S(w)')]
    KB3=[('On(tony,mike)',),('On(mike,john)',),('Green(tony)',),('~Green(john)',),('~On(xx,yy)','~Green(xx)','Green(yy)')]
    KB4 = [
('P(a,f(b))',), #1
('Q(c,y)',), #2
('~P(x,z)', '~Q(u,v)', 'R(x,u,f(z))'), #3
('~R(a,c,w)', 'S(w,b)'), #4
('~S(f(v),b)', 'T(v)'), #5
('~T(f(b))',), #6
('R(a,c,f(f(b)))',) #7
]
    KB5 = [
('F(f(a),g(b))',), #1
('G(h(c),k(d))',), #2
('~F(x,y)', '~G(u,v)', 'S(x,u)'), #3
('~S(w,z)', 'T(w,z)'), #4
('~T(f(a),h(c))',), #5
('S(f(a),h(c))',) #6
]
    
    res=solve(KB5)
    for i in range(len(res)):
        print(f'{i+1} {res[i]}')
                    
