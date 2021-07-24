import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("hello"+self.name+"!")

    def goodbye(self):
        print("Good Bye"+self.name+"!")



x = np.array([[1.00, 2.00, 3.00],[3,4,3]])
y = np.array([[2.00, 4.00, 6.00],[2,4,6]])
type(x)
print(x*10) #another ex for broadcast

#broadcast

A=np.array([[2,3],[4,5]])
B=np.array([10,20])

print(x[0][1])

for row in x:
    print(row)

A=A.flatten()
print(A)
print(A>4)


'''x=np.arange(0,6,0.1)
y=np.sin(x)
plt.plot(x,y,linestyle="dotted", label="sin")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin &cos')
plt.legend()
plt.show()'''

img = imread(r"C:\Users\Lucid\PycharmProjects\untitled2\kasumi.jpg")   #/U recognized unicode by computer, error occured
plt.imshow(img)
plt.show()


