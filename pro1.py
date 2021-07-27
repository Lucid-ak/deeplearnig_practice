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

def ReLU(x):
    return np.array(np.maximum(0, x))

def init_network():       #network 배열에 라벨링을 통해 각 가중치 및 편향 저장
    network={}
    network['W1']=np.array([0.1,0.3,0.5],[0.2,0.4,0.6])
    network['b1']=np.array([0.1, 0.2, 0.3])
    network['W2']=np.array([0.1,0.4],[0.2,0.5],[0.3,0.6])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([0.1,0.3],[0.2,0.4])
    network['b3']=np.array([0.1,0.2])

    return network

def forward(network, x):
    W1,W2, W3= network['W1'], network['W2'], network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y=identity_function(a3)

    return y

network=init_network()
x=np.array([1.0,0.5])
y=forward(network, x)
print(y)


x=np.arange(-5,5,0.1)          #최대 최소 간격 : 만큼의 numpy 배열을 형성
y=ReLU(x)

#print(y)

X=np.array([1.0,0.5])
W1=np.array([0.1,0.3,0.5],[0.2,0.4,0.6])
B1=np.array([0.1, 0.2, 0.3])

A1=np.dot(X,W1)+B1
Z1=sigmoid(A1)


#plt.plot(x,y)
#plt.ylim(-0.1,1.1)
#plt.show()

