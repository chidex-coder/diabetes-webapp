#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sysconfig import get_python_version
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_python_version().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


# loading the diabetes data to a pandas DataFrame
diabetes_data = pd.read_csv('diabetes.csv')


# In[3]:


# printing the first 5 rows of the data
diabetes_data.head()


# In[4]:


# printing the last 5 rows of the data
diabetes_data.tail()


# In[5]:


from dataprep.eda import plot
# using dataprep's plot method to get insights on each variable
plot(diabetes_data)


# In[6]:


from pandas_profiling import ProfileReport
# generate report with pandas profiling
profile = ProfileReport(diabetes_data, title='Report')
profile


# In[7]:


profile.to_file("report.html")


# In[8]:


import sweetviz as sv
my_report = sv.analyze(diabetes_data)
my_report.show_html()


# In[9]:


diabetes_data.columns


# In[10]:


# number of rows and Columns in this dataset
diabetes_data.shape


# In[11]:


diabetes_data.size


# In[12]:


diabetes_data.isnull()


# In[13]:


diabetes_data.isnull().sum()


# In[14]:


# getting the statistical measures of the data
diabetes_data.describe()


# In[15]:


diabetes_data['Outcome'].value_counts()


# In[16]:


diabetes_data.groupby('Outcome').mean()


# In[17]:


diabetes_data.duplicated().sum()


# In[18]:


diabetes_data.describe().T


# In[19]:


diabetes_data.head()


# In[20]:


diabetes_data.corr()


# In[21]:


## Visualization of Purchase with age
sns.scatterplot('Age','Pregnancies',hue='Outcome',data=diabetes_data);


# In[22]:


# separating the data and labels
X = diabetes_data.drop(columns = 'Outcome', axis=1)
Y = diabetes_data['Outcome']


# In[23]:


print(X)


# In[24]:


print(Y)


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, stratify=Y, random_state=0)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# In[27]:


classifier = svm.SVC(kernel='linear')


# In[28]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[29]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[30]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[31]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[32]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[33]:


input_data = (1,85,66,29,0,26.6,0.351,31) # 0

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('This patient is not diabetic')
else:
    print('This patient is diabetic')


# In[34]:


input_data = (0,137,40,35,168,43.1,2.288,33) # 1


# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('This patient is not diabetic')
else:
    print('This patient is diabetic')


# In[35]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train, Y_train)


# In[36]:


y_pred = logisticRegr.predict(X_test)
y_pred


# In[37]:


# Use score method to get accuracy of model
score = logisticRegr.score(X_test, Y_test)
print(score)


# In[38]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred, Y_test)


# In[39]:


y_pred.shape


# In[40]:


Y_test.shape


# In[41]:


from sklearn import metrics
cm = metrics.confusion_matrix(Y_test, y_pred)
print(cm)


# In[42]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[43]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[44]:


diabetes_data = RandomForestClassifier(random_state=1)
diabetes_data.fit(X_train, Y_train)


# In[45]:


diabetes_data_pred_train = diabetes_data.predict(X_train)
diabetes_data_pred_test = diabetes_data.predict(X_test)


# In[46]:


diabetes_data.score(X_train, Y_train)


# In[47]:


diabetes_data.score(X_test, Y_test)


# In[48]:


confusion_matrix(Y_train, diabetes_data_pred_train)
sns.heatmap(confusion_matrix(Y_train, diabetes_data_pred_train),annot=True, fmt='d',cbar=False, linewidths=.5, square = True, cmap='rainbow')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.show()


# In[49]:


confusion_matrix(Y_test, diabetes_data_pred_test)
sns.heatmap(confusion_matrix(Y_test, diabetes_data_pred_test),annot=True, fmt='d',cbar=False, linewidths=.5, square = True, cmap='rainbow')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.show()


# In[50]:


xgb_model = XGBClassifier(n_estimators=1000)
xgb_model.fit(X_train, Y_train, eval_metric='merror')


# In[51]:


from sklearn.metrics import accuracy_score, classification_report
diabetes_data_pred_train = xgb_model.predict(X_train)
diabetes_data_pred_test = xgb_model.predict(X_test)
target = sorted(set(Y))

print(f'Training accuracy: {accuracy_score(Y_train, diabetes_data_pred_train)}')
print(f'Training:\n {classification_report(Y_train, diabetes_data_pred_train, labels=target)}')
print(f'Testing accuracy: {accuracy_score(Y_test, diabetes_data_pred_test)}')
print(f'Testing:\n {classification_report(Y_test, diabetes_data_pred_test, labels=target)}')


# In[52]:


y_predicted = xgb_model.predict(X_test)
y_predicted


# In[53]:


print(len(Y_test))


# In[54]:


confusion_matrix(Y_test,y_predicted)


# In[55]:


confusion_matrix(Y_test, diabetes_data_pred_test)
sns.heatmap(confusion_matrix(Y_test, diabetes_data_pred_test),annot=True, fmt='d',cbar=False, linewidths=.5, square = True, cmap='rainbow')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.show()


# In[56]:


#import joblib
 
# Save the model as a pickle in a file
#joblib.dump(XGBClassifier(n_estimators=1000), 'xgb_model.pkl')


# In[57]:


# Load the model from the file
#xgb_from_joblib = joblib.load('xgb_model.pkl')


# In[58]:


# Use the loaded model to make predictions
#xgb_from_joblib.predict(X_test)


# In[59]:


import pickle
# Checks first to see if file already exists.
# If not the model is saved to disk
import os.path
if os.path.isfile('diabetes_data.sav') is False:
    pickle.dump(classifier, open('diabetes_data.sav', 'wb'))


# In[60]:


# loading the saved model
loaded_model = pickle.load(open('diabetes_data.sav', 'rb'))


# In[61]:


input_data = (6,148,72,35,0,33.6,0.627,50) # 1

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('This patient is not diabetic')
else:
    print('This patient is diabetic')


# In[62]:


input_data = (1,89,66,23,94,28.1,0.167,21) # 0

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('This patient is not diabetic')
else:
    print('This patient is diabetic')


# In[ ]:




