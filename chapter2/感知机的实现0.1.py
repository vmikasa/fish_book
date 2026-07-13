def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tem=x1*w1+x2*w2
    if tem>theta:
        return 1
    else:
        return 0

def OR(x1,x2):
    w1,w2,theta=0.5,0.5,0.3
    tem=x1*w1+x2*w2
    if tem>theta:
        return 1
    else:
        return 0

def NOT(x1):
    if x1>=0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    tem=AND(x1,x2)
    return NOT(tem)

def NOR(x1,x2):
    tem=OR(x1,x2)
    return NOT(tem)



y1=NAND(0,1)
print(y1)

# 偏置b，bias，描述重要程度。公式b+w1*x1+w2*x2<=0，b+w1*x1+w2*x2>0。显然，b越大，那么整个式子越容易大于0，这个神经元就越容易为1
