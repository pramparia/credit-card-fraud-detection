#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Credit Card Fraud Detection
#Ideas:
#Loading the dataset
#Checking the target classes
#Set target attribute & Split data into traing and test
#Resampling: Under-Sample
#4.a Logistic Regression w/o gridsearchcv
#4.b Decision Tree
#4.c Random Forest
#Resampling: Over-sample
#5.a Logistic Regression
#5.b Decision Tree
#5.c Random Forest
#Whole Data
#6.a Logistic Regression
#6.b Decision Tree
#6.c Random Forest


# In[10]:


#Bagging on Three Datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


#1.Loading the dataset
data = pd.read_csv('creditcard.csv')
print(data.shape)
data.head(3)


# In[21]:


#2. Checking the target classes
count_classes = pd.value_counts(data['Class']).sort_index()


# In[26]:


# Set target attribute & Split data into training and test
X = data.drop(['Time','Class'], axis=1)
y = data.loc[:, data.columns == 'Class']  # Using .loc instead of .ix
from sklearn.model_selection import train_test_split  # Changed from sklearn.cross_validation to sklearn.model_selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
print("Number of transactions train dataset: ", len(X_train))
print("Number of transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train) + len(X_test))

number_records_fraud = len(y_train[y_train.Class == 1])
print(number_records_fraud)
number_records_normal = len(y_train[y_train.Class == 0])
print(number_records_normal)


# In[24]:





# In[33]:


#4. Resampling: Under-Sample
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=33)
X_train_undersample, y_train_undersample = rus.fit_resample(X_train, y_train)  # Changed fit_sample to fit_resample
print(X_train_undersample.shape)
print(y_train_undersample.shape)

y = column_or_1d(y, warn=True)
count_classes = pd.value_counts(y_test['Class']).sort_index()


# In[31]:





# In[34]:


#4.a Logistic Regression - Recall
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report 


# In[36]:


lrclf = LogisticRegression()
lrclf = lrclf.fit(X_train_undersample,y_train_undersample)
train_under_p = lrclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_under_p))
test_under_p = lrclf.predict(X_test)
print(classification_report(y_test, test_under_p))


# In[38]:


param_grid = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100]}
lrclf = GridSearchCV(LogisticRegression(), param_grid,cv=5)
lrclf = lrclf.fit(X_train_undersample,y_train_undersample)
lrclf.best_params_
{'C': 1, 'penalty': 'l2'}
lrclf = LogisticRegression(C=1,penalty='l2')
lrclf = lrclf.fit(X_train_undersample,y_train_undersample)
train_under_p = lrclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_under_p))
test_under_p = lrclf.predict(X_test)
print(classification_report(y_test, test_under_p))


# In[39]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(y_test,test_under_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[41]:


#4.b Decision Tree
treeclf = DecisionTreeClassifier()
treeclf = treeclf.fit(X_train_undersample,y_train_undersample)
train_under_p = treeclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_under_p))
test_under_p = treeclf.predict(X_test)
print(classification_report(y_test, test_under_p))


# In[ ]:


param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(2,10), 'min_samples_split':np.arange(2,10),'min_samples_leaf':np.arange(1,10)}
treeclf = GridSearchCV(DecisionTreeClassifier(), param_grid,cv=5)
treeclf = treeclf.fit(X_train_undersample,y_train_undersample)
treeclf.best_params_
{'criterion': 'entropy',
 'max_depth': 3,
 'min_samples_leaf': 1,
 'min_samples_split': 4}
treeclf = DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=1,min_samples_split=4)
treeclf = treeclf.fit(X_train_undersample,y_train_undersample)
train_under_p = treeclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_under_p))
test_under_p = treeclf.predict(X_test)
print(classification_report(y_test, test_under_p))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_under_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

4.c Random Forest

rfclf = RandomForestClassifier()
rfclf = rfclf.fit(X_train_undersample,y_train_undersample)
train_under_p  = rfclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_under_p))
test_under_p = rfclf.predict(X_test)
print(classification_report(y_test, test_under_p))


# In[ ]:


param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(2,10), 'min_samples_split':np.arange(2,10),'min_samples_leaf':np.arange(1,10)}
rfclf = GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
rfclf = rfclf.fit(X_train_undersample,y_train_undersample)
rfclf.best_params_
{'criterion': 'gini',
 'max_depth': 5,
 'min_samples_leaf': 9,
 'min_samples_split': 2}
rfclf = RandomForestClassifier(criterion='gini', max_depth=5,min_samples_leaf=9,min_samples_split=2)
rfclf = rfclf.fit(X_train_undersample,y_train_undersample)
train_under_p = rfclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_under_p))
test_under_p = rfclf.predict(X_test)
print(classification_report(y_test, test_under_p))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_under_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

5. Resampling: Over-sample
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=33,svm_estimator=None)
X_train_oversample, y_train_oversample = sm.fit_sample(X_train,y_train)
print(X_train_oversample.shape)
print(y_train_oversample.shape)


# In[ ]:


number_records_fraud = len(y_train_oversample[y_train_oversample == 1])
print(number_records_fraud)
number_records_normal = len(y_train_oversample[y_train_oversample == 0])
print(number_records_normal)


# In[6]:


#5.a Logistic Regression
lrclf = LogisticRegression()
lrclf = lrclf.fit(X_train_oversample,y_train_oversample)
train_over_p = lrclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_over_p))
test_over_p = lrclf.predict(X_test)
print(classification_report(y_test, test_over_p))


# In[ ]:


param_grid = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100]}
lrclf = GridSearchCV(LogisticRegression(), param_grid,cv=5)
lrclf = lrclf.fit(X_train_oversample,y_train_oversample)
lrclf.best_params_
{'C': 100, 'penalty': 'l2'}
lrclf = LogisticRegression(penalty='l2' , C= 100)
lrclf = lrclf.fit(X_train_oversample,y_train_oversample)
train_over_p = lrclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_over_p))
test_over_p = lrclf.predict(X_test)
print(classification_report(y_test, test_over_p))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_over_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


#5.b Decision Tree

treeclf = DecisionTreeClassifier()
treeclf = treeclf.fit(X_train_oversample,y_train_oversample)
train_over_p = treeclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_over_p))
test_over_p = treeclf.predict(X_test)
print(classification_report(y_test, test_over_p))


# In[ ]:


param_grid = {'max_depth': np.arange(2,5), 'min_samples_split':np.arange(2,5),'min_samples_leaf':np.arange(1,5)}
treeclf = GridSearchCV(DecisionTreeClassifier(), param_grid,cv=5)
treeclf = treeclf.fit(X_train_oversample,y_train_oversample)
treeclf.best_params_
{'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2}
treeclf = DecisionTreeClassifier(max_depth=4,min_samples_leaf=1,min_samples_split=2)
treeclf = treeclf.fit(X_train_oversample,y_train_oversample)
train_over_p = treeclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_over_p))
test_over_p = treeclf.predict(X_test)
print(classification_report(y_test, test_over_p))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_over_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


#5.c Random Forest

rfclf = RandomForestClassifier()
rfclf = rfclf.fit(X_train_oversample,y_train_oversample)
train_p_over = rfclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_p_over))
test_p_over = rfclf.predict(X_test)
print(classification_report(y_test, test_p_over))


# In[ ]:


param_grid = {'max_depth': np.arange(2,5), 'min_samples_split':np.arange(2,5),'min_samples_leaf':np.arange(1,5)}
rfclf = GridSearchCV(RandomForestClassifier(), param_grid,cv=5)
rfclf = rfclf.fit(X_train_oversample,y_train_oversample)
rfclf.best_params_
{'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 4}
rfclf = RandomForestClassifier(max_depth=4,min_samples_leaf=1,min_samples_split=4)
rfclf = rfclf.fit(X_train_oversample,y_train_oversample)
train_p_over = rfclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_p_over))
test_p_over = rfclf.predict(X_test)
print(classification_report(y_test, test_p_over))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_over_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# In[ ]:


#6. Original data
#6.a Logistic Regression

param_grid = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100]}
lrclf = GridSearchCV(LogisticRegression(), param_grid,cv=5)
lrclf = lrclf.fit(np.array(X_train),np.ravel(y_train))
lrclf.best_params_
{'C': 10, 'penalty': 'l1'}
lrclf = LogisticRegression(penalty='l1',C=10)
lrclf = lrclf.fit(X_train,y_train)
train_all_p = lrclf.predict(X_train)
print(classification_report(y_train, train_all_p))
test_all_p = lrclf.predict(X_test)
print(classification_report(y_test, test_all_p))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_all_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

#6.b Decision Tree

param_grid = {'max_depth': np.arange(2,5), 'min_samples_split':np.arange(2,5),'min_samples_leaf':np.arange(1,5)}
treeclf = GridSearchCV(DecisionTreeClassifier(), param_grid,cv=5)
treeclf = treeclf.fit(np.array(X_train),np.ravel(y_train))
treeclf.best_params_
{'max_depth': 4, 'min_samples_leaf': 4, 'min_samples_split': 2}
treeclf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, min_samples_split=2)
treeclf = treeclf.fit(X_train,y_train)
train_all_p = treeclf.predict(X_train)
print(classification_report(y_train, train_all_p))
test_all_p = treeclf.predict(X_test)
print(classification_report(y_test, test_all_p))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_all_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

#6c. random forest

param_grid = {'max_depth': np.arange(2,5), 'min_samples_split':np.arange(2,5),'min_samples_leaf':np.arange(1,5)}
rfclf = GridSearchCV(RandomForestClassifier(), param_grid,cv=5)
rfclf = rfclf.fit(np.array(X_train),np.ravel(y_train))
rfclf.best_params_
{'max_depth': 4, 'min_samples_leaf': 1, 'min_samples_split': 2}
rfclf = RandomForestClassifier(max_depth=4,min_samples_split=2,min_samples_leaf=1)
rfclf = rfclf.fit(X_train,y_train)
train_all_p = rfclf.predict(X_train)
print(classification_report(y_train, train_all_p))
test_all_p = rfclf.predict(X_test)
print(classification_report(y_test, test_all_p))


# In[ ]:


cnf_matrix = confusion_matrix(y_test,test_all_p)
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

#7. Bagging on Three Datasets
Under sample
from sklearn.ensemble import BaggingClassifier
bcclf = BaggingClassifier()
bcclf = bcclf.fit(X_train_undersample,y_train_undersample)
train_p_under = bcclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_p_under))
test_p_under = bcclf.predict(X_test)
print(classification_report(y_test, test_p_under))


# In[ ]:


bcclf = BaggingClassifier(base_estimator=lrclf)
bcclf = bcclf.fit(X_train_undersample,y_train_undersample)
train_p_under = bcclf.predict(X_train_undersample)
print(classification_report(y_train_undersample, train_p_under))
test_p_under = bcclf.predict(X_test)
print(classification_report(y_test, test_p_under))


# In[ ]:


bcclf = BaggingClassifier()
bcclf = bcclf.fit(X_train_oversample,y_train_oversample)
train_p_over = bcclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_p_over))
test_p_over = bcclf.predict(X_test)
print(classification_report(y_test, test_p_over))


# In[ ]:


bcclf = BaggingClassifier(base_estimator=lrclf)
bcclf = bcclf.fit(X_train_oversample,y_train_oversample)
train_p_over = bcclf.predict(X_train_oversample)
print(classification_report(y_train_oversample, train_p_over))
test_p_over = bcclf.predict(X_test)
print(classification_report(y_test, test_p_over))


# In[ ]:


bcclf = BaggingClassifier()
bcclf = bcclf.fit(X_train,y_train)
train_p_all = bcclf.predict(X_train)
print(classification_report(y_train, train_p_all))
test_p_all = bcclf.predict(X_test)
print(classification_report(y_test, test_p_all))

