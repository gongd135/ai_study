#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install -U scikit-learn')


# In[5]:


from sklearn import datasets
iris = datasets.load_iris()
print(iris)


# In[8]:


from sklearn.model_selection import train_test_split

X = iris.data
y = iris.target
#(80:20)으로 테스트 데이터와 실행 데이터로 분할

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# In[31]:


y_pred = knn.predict(X_test)
#knn을 이용해서 input부분만 이용해서 x_test가 어떤 값을 가지는지 확인
from sklearn import metrics

scores = metrics.accuracy_score(y_test, y_pred) 
#테스트 정확도 확인
print(scores)


# In[33]:


classes = {0:'setosa', 1:'versicolor', 2:'virginica'} #클래스의 이름들을 지정

x_new = [[3,4,5,2], [5,4,2,2], [3.1,3.9, 4.5, 2.3]] #새로운 샘플 넣기

y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])
print(classes[y_predict[2]])


# In[ ]:




