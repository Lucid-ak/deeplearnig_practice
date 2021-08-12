import pickle

import numpy as np          #비선형 퍼셉트론
import matplotlib.pylab as plt
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

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

def softmax(x):     #c는 입력 값 중 최대
    c=np.max(x)
    exp_x=np.exp(x-c)
    sum_exp_x=sum(exp_x)

    y = exp_x/sum_exp_x      #return exp_x/sum_exp_x로 바로 나타낼 수도 있지만 "가시성"을 위해 y로 따로 배정하여 계산
    return y

def identity_function(x):
    return x

def init_network():       #network 배열에 라벨링을 통해 각 가중치 및 편향 저장
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1, 0.2, 0.3])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])

    return network

def init_networrk_mnist():         #라이브러리에서 weight, bias 가져오기
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1,W2, W3= network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y=softmax(a3)

    return y


def forward(network, x):      #순방향(입력->출력) 구현   항상 비슷한 값을 도출한다.
    W1,W2, W3= network['W1'], network['W2'], network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1)+b1
    z1 = softmax(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = softmax(a2)
    a3 = np.dot(z2, W3) + b3
    y=identity_function(a3)

    return y

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def mean_squared_error(t,y):
    return np.sum(np.square(y-t))*0.5 #/2안하는 이유는 float으로 바로 변환시키려고?

def cross_entropy_error(t,y):
    d=1e-7
    return -np.sum(t*np.log(y+d))           #만약 이대로 쓰면 로그 함수에 0넣게 되서 -무한대가 된다. 계산 불가 그러므로 최소한의 수를 둔다.

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size=10
batch_mask = np.random.choice(train_size, batch_szie)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
                                                       #ndim

t=[0,0,1,0,0,0,0,0,0,0]
y=[0.1,0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(t),np.array(y)))     #이거 반드시 np.array로 변환 해주어야 한다.
print(cross_entropy_error(np.array(t), np.array(y)))
print(np.array(t))
print(t)