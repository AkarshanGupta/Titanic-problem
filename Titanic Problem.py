#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test[("PassengerId")]

def clean(data):
    data = data.drop(["Ticket","Cabin","Name","PassengerId"], axis=1)
    # Fill missing values with median for numerical columns
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    
    # Fill missing values in the 'Embarked' column with 'U'
    data.Embarked.fillna("U", inplace=True)
    
    return data

data = clean(data)
test = clean(test)


# In[3]:


data.head(5)


# In[4]:


from sklearn import preprocessing
ln = preprocessing.LabelEncoder()

cols = ["Sex", "Embarked"]
for col in cols:
    data[col] = ln.fit_transform(data[col])
    test[col] = ln.transform(test[col])
    print(ln.classes_)

    
    
data.head(5)  


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

Y= data["Survived"]
X = data.drop("Survived",axis=1)

X_train, X_val,Y_train,Y_val = train_test_split(X,Y, test_size=0.2, random_state=42)


# In[6]:


clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, Y_train)


# In[7]:


predictions = clf.predict(X_val)
from sklearn.metrics import accuracy_score 
accuracy_score(Y_val, predictions)


# In[8]:


submission_preds = clf.predict(test)


# In[9]:


df = pd.DataFrame({"PassengerId":test_ids.values,
                  "Survived":submission_preds,
                  })


# In[10]:


df.to_csv("submission.csv", index = False)

