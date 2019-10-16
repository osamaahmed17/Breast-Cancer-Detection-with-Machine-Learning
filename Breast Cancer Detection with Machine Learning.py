import sys
import numpy
import matplotlib
import pandas
import sklearn

print('Python:',format(sys.version))
print('Numpy: ',format(numpy.__version__))
print('Matplotlib:',format(matplotlib.__version__))
print('Pandas: ',format(pandas.__version__))
print('Sklearn:',format(sklearn.__version__))

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier #K nearest  (It tries to cluster different datapoints into groups)
from sklearn.svm import SVC #Support Vector Machine (Looking for hyper optimal hyperplane which can seperate these datapoints into cancer and non cancer cells)
from sklearn.model_selection import KFold
from sklearn import model_selection #So both of the models could be used at the same time
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix #For Plotting
import matplotlib.pyplot as plt
import pandas as pd

#Loading the Data from UCI reporsitory
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names=['id','clump_thickness','univorm_cell_size','univorm_cell_shape',
       'marginal_adhesion','single_epithelial_size','bare_nuclei',
       'bland_chromatin','normal_nuclei','mitoses','class']
df=pd.read_csv(url,names=names)

#Preprocessing the Data 
df.replace('?',-99999,inplace=True)
print(df.axes)

#We are dropping the ID column because it couldn't be used for a classifier
df.drop(['id'],1,inplace=True)

#Printing the shape of the DataFrame


df.dropna(inplace=True)

print(df.shape)

#Do Dataset Visualization
print(df.loc[0])
print(df.describe())

#Plot Histograms for each Variables
df.hist(figsize=(10,10)) #For making visualization look good
plt.show()

#Create Scatter Plot Matrix
scatter_matrix(df,figsize=(18,18))
plt.show()

#Creat X and Y dataset for Training
X=np.array(df.drop(['class'],1))
Y=np.array((df['class']))


X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Specify Testing Option
seed= 8
scoring='accuracy'

#Define the Model to Train
models=[]
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM',SVC()))

#Evaluate each model in Turn
results=[]
names=[]
for name, model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name,cv_results.mean(),cv_results.std())
    print(msg)

#Make prediction on Validation Dataset
for name, model in models:
    model.fit(X_train,Y_train)
    predictions=model.predict(X_test)
    print(name)
    print(accuracy_score(Y_test,predictions))
    print(classification_report(Y_test,predictions))

#Precisions: A ratio of correctly predicted postive observations over the total predicted postive observations.
#High precision means that we dont have many false positive

#Recall: Its the measure of false negative

#For testing our model on an example set
clf = SVC()
clf.fit(X_train,Y_train)
accuracy=clf.score(X_test,Y_test)
print(accuracy)

example=np.array([[4,2,1,1,1,2,3,2,1]])
example=example.reshape(len(example),-1)
prediction=clf.predict(example)
print(prediction)