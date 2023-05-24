#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
epsilon = 0

def perceptron(x1, x2):
    X = np.array([x1, x2])
    W = np.array([1.0, 1.0])
    B = -1.5
    sum = np.dot(W, X) +B
    if sum > epsilon:
        return 1
    else:
        return 0
    
print(perceptron(0, 0))
print(perceptron(1, 0))
print(perceptron(0, 1))
print(perceptron(1, 1))


# In[56]:


import numpy as np

epsilon = 0 #부동소수점 오차 방지

def step_func(t): #퍼셉트론 활성화 함수
    if t > epsilon:
        return 1
    else:
        return 0
                    
                    #훈련 데이터 셋
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([0,0,0,1]) #정답값
W = np.zeros(len(X[0])) #가중치 저장용 행렬


# In[57]:


def perceptron_fit(X,Y, epochs=10):#input: X, 라벨값: Y, epochs: 학습할 횟수
    global W
    eta = 0.2 #학습률
    
    for t in range(epochs):
        print("epoch=", t ,"=========================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W)) #input값과 가중치(weight)값을 곱해줌
            error = Y[i] - predict #error: 정답과 predict간의 오차
            W += eta * error * X[i] #가중치를 업데이트 해줌
            print("현재 처리 입력 =", X[i],"정답 =", Y[i], "출력 =", predict, "변경된 가중치 =", W)
        print("=====================================")


# In[58]:


def perceptron_predict(X, Y): #예측 부분
    global W
    for x in X:
        print(x[0], x[1], "->", step_func(np.dot(x, W))) #x가 들어갔을떄 어떤 값이 나오는지
        
perceptron_fit(X, y, 6)
perceptron_predict(X, y)


# In[59]:


import numpy as np

epsilon = 0 #부동소수점 오차 방지

def step_func(t): #퍼셉트론 활성화 함수
    if t > epsilon:
        return 1
    else:
        return 0
                    
                    #훈련 데이터 셋
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

y = np.array([0,0,0,1]) #정답값
W = np.zeros(len(X[0])) #가중치 저장용 행렬

def perceptron_fit(X,Y, epochs=30):#input: X, 라벨값: Y, epochs: 학습할 횟수
    global W
    eta = 0.1 #학습률
    
    for t in range(epochs):
        print("epoch=", t ,"=========================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i], W)) #input값과 가중치(weight)값을 곱해줌
            error = Y[i] - predict #error: 정답과 predict간의 오차
            W += eta * error * X[i] #가중치를 업데이트 해줌
            print("현재 처리 입력 =", X[i],"정답 =", Y[i], "출력 =", predict, "변경된 가중치 =", W)
        print("=====================================")
        
def perceptron_predict(X, Y): #예측 부분
    global W
    for x in X:
        print(x[0], x[1], "->", step_func(np.dot(x, W))) #x가 들어갔을떄 어떤 값이 나오는지
        
perceptron_fit(X, y, 10)
perceptron_predict(X, y)


# In[1]:


from sklearn.linear_model import Perceptron

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]

clf = Perceptron(tol=1e-3, random_state = 0)

clf.fit(X, y)

print(clf.predict(X))


# In[3]:


from sklearn.linear_model import Perceptron

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0] #xor연산을 구현 하려고 y 레이어를 xor연산답으로 설정

clf = Perceptron(tol=1e-3, random_state = 0)

clf.fit(X, y)

print(clf.predict(X)) #xor은 싱글 레이어로 구현이 안됨


# In[ ]:




