
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture 

df = pd.read_csv("seismic_bumps1.csv").drop(['nbumps','nbumps2','nbumps3','nbumps4','nbumps5','nbumps6','nbumps7','nbumps89'],axis=1)

#QUES1
print("\n\n*******QUESTION-1*******")
y=df["class"]
X=df.drop("class",axis=1)
[X_train,X_test,y_train,y_test]=train_test_split(X,y, test_size=0.3, random_state=42,shuffle=True)
pd.DataFrame(X_train).to_csv("seismic-bumps-train.csv",index=None)
pd.DataFrame(X_test).to_csv("seismic-bumps-test.csv",index=None)
accuracy=[]

def classify(n,xtrain,xtest):
    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(xtrain,y_train)
    y_pred = classifier.predict(xtest)
    accuracy.append(100*accuracy_score(y_test, y_pred))
    print("(a)Confusion Matrix: \n",confusion_matrix(y_test, y_pred))                  #part(a)
    print("(b)Classification Accuracy: ",100*accuracy_score(y_test, y_pred),"%")       #part(b)

print("\nClassification using K-NN method:")
print("\nFor K=1 :-");classify(1,X_train,X_test)
print("\nFor K=3 :-");classify(3,X_train,X_test)
print("\nFor K=5 :-");classify(5,X_train,X_test)
print("\nAccuracy is highest for K=5")

#QUES2
print("\n\n*******QUESTION-2*******")
X_train_minmax = MinMaxScaler().fit_transform(X_train)
X_test_minmax = MinMaxScaler().fit_transform(X_test)
pd.DataFrame(X_train_minmax).to_csv("seismic-bumps-train-Normalised.csv",index=None,header=None)
pd.DataFrame(X_test_minmax).to_csv("seismic-bumps-test-normalised.csv",index=None,header=None)

print("\nClassification using K-NN method on normalized data:")
print("\nFor K=1 :-");classify(1,X_train_minmax,X_test_minmax)
print("\nFor K=3 :-");classify(3,X_train_minmax,X_test_minmax)
print("\nFor K=5 :-");classify(5,X_train_minmax,X_test_minmax)
print("\nAccuracy is highest for K=5")

#QUES3
print("\n\n*******QUESTION-3*******")
gmm = GaussianMixture(n_components = 1)
gmm.fit(X_train)
y_bayes=gmm.predict(X_test)

pd.options.display.max_columns = None
print("\nClassification using Bayes Classifier using unimodal Gaussian density: ")
print("Confusion Matrix: \n",confusion_matrix(y_test,y_bayes ))          
print("Classification Accuracy: ",100*accuracy_score(y_test, y_bayes),"%")     

print("\nFor class = 0 :-")
mean_0=pd.DataFrame.mean(df.where(df["class"]==0).drop("class",axis=1))
print("Mean vector: \n",mean_0)
cov_0=pd.DataFrame.cov(df.where(df["class"]==0).drop("class",axis=1))
print("\nCovariance Matrix: \n",cov_0)

print("\nFor class = 1 :-")
mean_1=pd.DataFrame.mean(df.where(df["class"]==1).drop("class",axis=1))
print("Mean vector: \n",mean_1)
cov_1=pd.DataFrame.cov(df.where(df["class"]==1).drop("class",axis=1))
print("\nCovariance Matrix: \n",cov_1)

#QUES4
print("\n\n*******QUESTION-4*******")
l=[accuracy[2],accuracy[-1],100*accuracy_score(y_test, y_bayes)]

table=pd.DataFrame(l,["K-NN Classifier","K-NN Classifier(normalized data)","Bayes Classifier(unimodal Gaussian density)"],["Accuracy(in %)"])
print(table)
