import numpy as np          #비선형 퍼셉트론
import matplotlib.pylab as plt

def AND(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b= -0.7
    theta = 0
    tmp = np.sum(w*x)+b
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1
def OR(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.3
    theta = 0
    sig=np.sum(w*x)+b
    if sig>=theta:
        return 1
    else :
        return 0
def NAND(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.7
    theta=0
    sig=np.sum(w*x)+b
    if sig<theta :
        return 1
    else :
        return 0
def XOR(x1, x2):
    y1=OR(x1,x2)
    y2=NAND(x1,x2)
    y=AND(y1,y2)
    return y
def step_function(x):
    '''
    y = x > 0
    return y.astype(np.int)                     #np.int와 dtype=int의 역할은 같다.
    '''
    return np.array(x>0, dtype=int)             #dtype의 역할은 출력 결과를 dtype=int등으로 통해 원하는 자료형으로 변형하는 것

def sigmoid(x):
    return 1/(1+np.exp(-x))      #브로드 캐스트 적용, 각 배열의 원소값에 대해 계산 후 결과값들을 배열로 변환

'''
print(AND(0,0))
print(AND(1,0))
print(AND(0,1))
print(AND(1,1))

print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

print(NAND(0,0))
print(NAND(1,0))
print(NAND(0,1))
print(NAND(1,1))

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))
'''
x=np.arange(-5,5,0.1)          #최대 최소 간격 : 만큼의 numpy 배열을 형성
y=sigmoid(x)

print(y)

plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

