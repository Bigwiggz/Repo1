
############################################################################## 
#Classifier Example
##############################################################################

#Wine Classifier

#Import all libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn import MLPClassifier
#form sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
%matplotlib inline

#--------------------

#Loading dataset
wine=pd.read_csv('winequality-red.csv',sep=';')

#--------------------

#preview dataset
wine.head()

#--------------------

#preview dataset
wine.info()

#--------------------

#look at null values
wine.isnull().sum()

#--------------------

#Bin Preprocessing
bins=(2,6.5,8)
group_names=['bad','good']
wine['quality']=pd.cut(wine['quality'], bins=bins, labels=group_names)
wine['quality'].unique()

#--------------------

label_quality=LabelEncoder()
wine['quality']=label_quality.fit_transform(wine['quality'])

#--------------------

#check labels-quality
wine.head(5)

#--------------------

wine['quality'].value_counts()

#--------------------

#plot wine bins
sns.countplot(wine['quality'])

#--------------------

#Now seperate dataset into test and training dataset
X=wine.drop('quality', axis=1)
y=wine['quality']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#--------------------

#For classifiers, you want to apply a standard scaler
#Applying Standard scaling to get optimized result
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#--------------------

Show last 10 rows
X_train[:10,:]


#Random Forest Classifier
##############################################################################
#Workcase-works best with medium sized data sets

rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc=rfc.predict(X_test)
pred_rfc[:20,:]

#--------------------

#Lets see how the rfc performed
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

#SVM Classifier
##############################################################################
#Workcase-works better on smaller numbers


svmc=svm.SVC()
svmc.fit(X_train, y_train)
pred_svmc=svmc.predict(X_test)
pred_svmc[:20,:]

#--------------------

#Lets see how the rfc performed
print(classification_report(y_test, pred_svmc))
print(confusion_matrix(y_test, pred_svmc))


#Neural Network
##############################################################################
#Workcase-works better on huge amounts of data
#With more layers you need to run more interations and there is the possibility of overfitting

mplc=MLPClassifier(hidden_layer_sizes=(11,11,11), max_inter=500)
mplc.fit(X_train, y_train)
pred_mplc=mplc.predict(X_test)
pred_mplc[:20,:]

#--------------------

#Lets see how the rfc performed
print(classification_report(y_test, pred_mplc))
print(confusion_matrix(y_test, pred_mplc))


##############################################################################
# Get Model Accuracy
##############################################################################


from sklearn.metrics import accuracy_score
rfc_acc=accuracy_score(y_test, pred_rfc)
rfc

wine.head(10)

#-------------------

#Test any new X value
X_new[[7.3,0.58,0.00,2.0,0.065,15.0,21.0,0.9946,3.36,0.47,10.0]]
X_new=sc.transform(X_new)
y_new=rfc.predict(X_new)
y_new


##############################################################################
# General Notes
##############################################################################

#MinMaxScaler 0-1 is good for regression, neural networks, and gradient decent

#StandardScaler is good for features that assume a normal distribution of the input variables and for classification_report

#Normalization is good for datasets with lots of 0s and with attributes of varying scales.  Like kNN.  normalization is done on each obersvation(row).

