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

'''
network=init_network()

x=np.array([100,40])
y=forward(network, x)
print(y)



#print(y)



#plt.plot(x,y)
#plt.ylim(-0.1,1.1)
#plt.show()
'''
x_test,t_test = get_data()
network=init_networrk_mnist()

batch_size=100
accuracy_ct=0

for i in range(0, len(x_test), batch_size):#x_train의 실제 개수 몰라도 len함수 쓰면 된다
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) #가장 확률이 높은 원소 가져오기?
    print(np.sum(p == t_test[i:i+batch_size]))
    accuracy_ct += np.sum(p == t_test[i:i+batch_size])

    '''
    y=predict(network, x_test[i])
    p=np.argmax(y)

    if p==t_test[i] :
        accuracy_ct+=1
    '''
print("Accuracy:",str(float(accuracy_ct)/len(x_test)))
print(accuracy_ct)
'''
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28) #이거 패턴화 28,28로 안하면 원하는 이미지 안나온다.-> 이거 이용해서 암호화나 용량 줄이기도 가능?
print(img.shape)

img_show(img)
'''
