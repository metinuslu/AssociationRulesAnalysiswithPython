
# coding: utf-8

# In[ ]:


get_ipython().system('python --version')


# In[ ]:


#Show the System Variables

import os
print(os.getcwd())


# In[ ]:


#Importing Modules
import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import mlxtend as ml
#print(ml.__version__)


# In[ ]:


# dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#            ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#            ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
#            ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
#            ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


# In[ ]:


dataset = []


# In[ ]:


import random

items = ['Apple','Corn','Dill','Eggs',
         'Ice cream','Kidney Beans','Milk','Nutmeg',
         'Onion','Unicorn','Yogurt', 'Bread', 
         'Cheese', 'Butter', 'Sugar','Chocolate']

for i in range(1, 1000):
    k = random.randrange(1, len(items)+1)
    #print('K Values:', k)
    dataset.append(random.choices(items, k=k))
    #print(random.sample(items, k=k))
    
    #liste.append(random.sample(items, k=k))
    #print(random.sample(items, k=k))


# In[ ]:


print(len(dataset))


# In[ ]:


print(dataset[1:5])


# In[ ]:


print(dataset[995:(len(dataset)+1)])


# In[ ]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)


# In[ ]:


df.head()


# In[ ]:


df.to_csv('./output/dataset.csv', index=False)


# In[ ]:


apriori(df, min_support=0.15)[1:26]


# In[ ]:


print("Kural Sayısı:", len(apriori(df, min_support=0.15)))


# In[ ]:


apriori(df, min_support=0.15, use_colnames=True)[1:26]


# In[ ]:


frequent_itemsets = apriori(df, min_support=0.15, use_colnames=True)

frequent_itemsets


# In[ ]:


association_rules(frequent_itemsets, metric="confidence", min_threshold=0.30)

rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.30)


# In[ ]:


print("Oluşan Kural Sayısı:", len(rules1))


# In[ ]:


rules1 = rules1.sort_values(['confidence'], ascending=False)
rules1[1:11]


# In[ ]:


rules1["antecedent_len"] = rules1["antecedents"].apply(lambda x: len(x))
rules1["consequents_len"] = rules1["consequents"].apply(lambda x: len(x))
rules1[1:6]


# In[ ]:


rules2 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules2 = rules2.sort_values(['lift'], ascending=False)

rules2[1:6]


# In[ ]:


rules2["antecedent_len"] = rules2["antecedents"].apply(lambda x: len(x))
rules2["consequents_len"] = rules2["consequents"].apply(lambda x: len(x))
rules2


# In[ ]:


rules1[(rules1['antecedent_len'] >= 1) &
       (rules1['confidence'] >= 0.20) &
       (rules1['lift'] > 1) ].sort_values(['confidence'], ascending=False)[1:10]


# In[ ]:


rules1[rules1['antecedents'] == {'Bread'}].sort_values(['confidence'], ascending=False)[1:10]


# In[ ]:


rules1.to_json('./output/rules1.json')
rules2.to_json('./output/rules2.json')

