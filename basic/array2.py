#!/usr/bin/env python
# coding: utf-8

# In[1]:


3.14*10*10


# In[2]:


3.14*10**2


# In[3]:


type(10)


# In[4]:


car = {'hp':200, 'make': "Ben"}


# In[6]:


car['hp']


# In[7]:


car['color']="white"


# In[8]:


car


# In[9]:


temp = -10


# In[12]:


if temp >= 0:
    print("영상입니다")
else:
    print("영하입니다")


# In[17]:


for i in [1,2,3,4,5]:
    print(i, end=",")


# In[18]:


def sayHello():
    print("Hello")


# In[20]:


sayHello()


# In[1]:


def sayHello(name):
    print("Hello!" + name)


# In[2]:


sayHello("Kim")


# In[31]:


class Person:
    def _init_(self, name, age):
        self.name = name
        self.age = age
        
    def sayHello(self):
        print("Hello 나의 이름은" + self.name)

            
p1 = Person("John", 38, 180)
p1.sayHello()


# In[32]:


import numpy as np


# In[33]:


a = np.array([1,2,3])


# In[34]:


a


# In[35]:


a[1]


# In[36]:


b = np.array([[1,2,3], [4,5,6], [7,8,9]])


# In[37]:


b


# In[39]:


b[0][2]


# In[2]:


import numpy as np
a = np.array([[0,1,2],[3,4,5],[6,7,8]])


# In[3]:


a.shape


# In[4]:


a.ndim


# In[6]:


a.dtype


# In[7]:


a.size


# In[8]:


a=np.zeros((3,4))


# In[9]:


a


# In[10]:


e=np.zeros((3,4,5))


# In[11]:


e


# In[12]:


np.ones(a.shape)


# In[13]:


a


# In[14]:


np.eye(3)


# In[15]:


np.eye(a.shape)


# In[16]:


np.linspace(0,10,10)


# In[17]:


np.linspace(0,10,100)


# In[18]:


np.linspace(0,10,99)


# In[19]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])


# In[20]:


x


# In[22]:


np.concatenate((x,y), axis=1)


# In[25]:


np.concatenate((x,y), axis=1)


# In[27]:


np.vstack((x,y))


# In[ ]:




