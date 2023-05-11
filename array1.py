#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a = np.array([1,2,3,4,5,6])


# In[3]:


a


# In[4]:


b = a.reshape((2,3))


# In[5]:


b


# In[6]:


a = np.arange(12)


# In[7]:


a


# In[8]:


a.reshape(3,4)


# In[11]:


a.reshape(6,-2)


# In[14]:


b = np.arange(30).reshape(-1, 10)


# In[15]:


b


# In[17]:


b1, b2 = np.split(b,[3],axis=1)


# In[18]:


b1


# In[19]:


b2


# In[20]:


a = np.array([1,2,3,4,5,6])


# In[21]:


a.shape


# In[22]:


b = np.array([7,8,9,10,11,12])


# In[23]:


b.shape


# In[24]:


ab = a[np.newaxis,:]


# In[25]:


ab


# In[26]:


ab = b[np.newaxis,:]


# In[27]:


ab


# In[28]:


ab = a,b[np.newaxis,:]


# In[29]:


ab


# In[31]:


ab = a[:,np.newaxis]


# In[32]:


ab


# In[33]:


a[1:3]


# In[34]:


c = a > 3


# In[35]:


c


# In[37]:


c=[a>3]


# In[38]:


c


# In[39]:


d = [a>3]


# In[40]:


d


# In[41]:


a


# In[42]:


d = None


# In[43]:


d


# In[46]:


c = a > 3


# In[47]:


c[c>3]


# In[48]:


c


# In[49]:


d[c>3]


# In[52]:


a[a>3]


# In[53]:


a = np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[54]:


a[0,2]


# In[55]:


a[0,0]


# In[56]:


a1 = np.array([[1,2],[3,4],[5,6]])
a2 = np.array([[1,1],[1,1],[1,1]])


# In[57]:


result = a1 + a2


# In[58]:


result


# In[64]:


a2 = np.array([[2,2],[2,2],[2,2]])


# In[65]:


result = a1 * a2


# In[66]:


result


# In[67]:


a1 = np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[68]:


result = a1 @ a2


# In[69]:


result


# In[73]:


a1.mean(axis=1)


# In[3]:


import numpy as np
np.random.seed(100)
np.random.rand(5)


# In[4]:


np.random.rand(5,3)


# In[5]:


np.random.randn(5)


# In[6]:


np.random.rand(5,4)


# In[7]:


mu, sig = 0, 0.1


# In[9]:


np.random.normal(mu, sig, 5)


# In[11]:


np.random.normal(mu, sig, (5,3))


# In[13]:


a1 = np.array([[1,2],[3,4],[5,6]])


# In[14]:


print(a1.T)


# In[16]:


a1.flatten()


# In[22]:


import matplotlib.pyplot as plt

X = ["mon","tue","wen","thur","fri","sat","sun"]
Y1 = [1.1,2.1,3.1,4.1,5.1,6.1,7]
Y2 = [7.1,8.1,9.1,10.1,11.1,12.1,13]

plt.plot(X, Y1, label="seoul")
plt.plot(X, Y2, label="busan")
plt.xlabel("day")
plt.xlabel("temp")
plt.legend(loc="upper left")
plt.title("Temperature")
plt.show()


# In[23]:


plt.plot(X, Y1, "sm")
plt.show()


# In[24]:


Xplt = np.arange(0,10)
Yplt = Xplt ** 2
plt.plot(Xplt, Yplt)
plt.show()


# In[ ]:




