import random
def Index(literal_idx,clause_idx,len):
    if len==1:
        return str(clause_idx+1)
    else:
        return str(clause_idx+1)+chr(ord('a')+literal_idx)
def iscomplementary(x,y):
    if not x or not y:  
        return False
    if x[0]=='~' and x[1:]==y[:]:
        return True
    if y[0]=='~' and y[1:]==x[:]:
        return True
    return False
def resolve(clause1,clause2,idx1,idx2):   #归结得到新子句
    newclause=list(clause1)+list(clause2)
    newclause.remove(clause1[idx1])
    newclause.remove(clause2[idx2])
    newclause=list(dict.fromkeys(newclause))
    return tuple(newclause)
def sequence(newclause,idx1,idx2):
    ans='R['+idx1+','+idx2+']='
    ans+=str(newclause)
    return ans
def contains_unordered(newclauset, newclause):
    newclause_set = set(newclause)
    for clause in newclauset:
        clause_set = set(clause)
        if clause_set == newclause_set:
            return True
    return False
def resolution(KB):
    ALL=KB.copy()
    support_list=[(ALL[-1],len(ALL)-1)]
    result=[]
    vis=set()

    newall=[]
    for i,x in enumerate(ALL,0):
        newall.append((x,i))
    idx=len(newall)
    while True:
        newclauset=[]
        newall=sorted(newall,key=lambda x:len(x[0]))
        for clause1,clause1_idx in newall:
            for clause2,clause2_idx in newall:
                if clause1_idx==clause2_idx :
                    continue
                if (clause1,clause2) in vis:
                    continue
                sup_list=[x[0] for x in support_list]
                if clause2 not in sup_list and clause1 not in sup_list:
                    continue
                for literal_idx1 in range(len(clause1)):
                    for literal_idx2 in range(len(clause2)):
                        literal1,literal2=clause1[literal_idx1],clause2[literal_idx2]
                        if not iscomplementary(literal1,literal2):
                            continue
                        '''处理互补对'''
                        newclause=resolve(clause1,clause2,literal_idx1,literal_idx2)
                        newcset=[x[0] for x in newclauset]
                        if contains_unordered(ALL,newclause) or contains_unordered(newcset,newclause):
                            continue
                        vis.add((clause1,clause2))
                        idx1=Index(literal_idx1,clause1_idx,len(clause1))
                        idx2=Index(literal_idx2,clause2_idx,len(clause2))
                        seq=sequence(newclause,idx1,idx2)
                        result.append(seq)

                        newclauset.append((newclause,idx))
                        idx+=1
                        if newclause==():
                            return result
        newall.extend(newclauset)
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
    res=list(KB.copy())+resolution(KB.copy())
                    
    res=simplify(res,len(KB.copy()))
    return res
if __name__=='__main__':
    KB = [
    ('A', 'B'),
    ('~A', 'C'),
    ('~B', 'D'),
    ('~C', 'E'),
    ('~D', 'F'),
    ('~E',),
    ('~F',)
]
    res=solve(KB)
    for i in range(len(res)):
        print(f'{i+1} {res[i]}')

                    
