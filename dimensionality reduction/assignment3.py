# Name- Akanksha Sinha
# Roll no.- B19125
# Mobile no.- 9284606382, 9928960217(Whatsapp)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("landslide_data3.csv")

#QUES1
print("\n\n*******QUESTION-1*******")
df1 = df.iloc[:,-7:]       #dataframe containing only last 7 attributes

#Replacing outliers by median of respective attributes
j=0
for i in df1.columns:
    Q1=np.quantile(df1[i],.25)
    Q3=np.quantile(df1[i],.75)
    outliers=df1[i][(df1[i]<Q1-1.5*(Q3-Q1)) | (df1[i]>Q3+1.5*(Q3-Q1))] #Detecting outliers
    df1.iloc[outliers.index,j]=(df1[i].drop(outliers.index)).median() #Replacing outliers with mean of data excluding outliers
    j+=1

#(a)
print("\n(a)")
df_norm = df1.copy()
#min-max normalization
for i in df_norm.columns:
    df_norm[i] = ((df_norm[i]-df_norm[i].min())/(df_norm[i].max()-df_norm[i].min()))*(9-3)+3

print("\nMin-Max values before normalization: ")
print(df1.agg([min,max]).T)
print("\nMin-Max values after normalization: ")
print(df_norm.agg([min,max]).T)

#(b)
print("\n(b)")
df_stand = df1.copy()
#standardization
for i in df_stand.columns:
    df_stand[i] = (df_stand[i]-df1.mean()[i])/df1.std()[i]

print("\nMean and standard deviation before Standardization: ")
print(pd.concat([df1.mean(),df1.std()],keys=["Mean","Standard Deviation"],axis=1))
print("\nMean and standard deviation after Standardization: ")
print(pd.concat([df_stand.mean(),df_stand.std()],keys=["Mean","Standard Deviation"],axis=1))

#QUES2
print("\n*******QUESTION-2*******")
mean=[0,0]
cov=[[5,10],[10,13]]
D = np.random.multivariate_normal(mean, cov, 1000)
x,y = np.random.multivariate_normal(mean, cov, 1000).T

#(a)
print("\n(a)")
plt.scatter(x,y,marker='x',c='blue')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Plot of 2D synthetic data") #scatter plot of data samples
plt.show()

#(b)
print("\n(b)")
eigval, eigvec = np.linalg.eig(np.cov(D.T))
print("Eigenvalues: ",eigval)
print("Eigenvectors: \n",eigvec)
origin=[0,0]
plt.scatter(x,y,marker='x',c='blue')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Plot of 2D synthetic data and eigen directions")
plt.quiver(*origin,*eigvec[:,0],color="red",scale=7) #Eigen direction1
plt.quiver(*origin,*eigvec[:,1],color="red",scale=3) #Eigen direction2
plt.show()

#(c)
print("\n(c)")
def projection(d,v):
    a=[]
    for i in range(len(d)):
        proj=(np.dot(d[i],v)/(np.dot(v,v)))*v
        a.append(proj.tolist())
    return(np.array(a))

a1=projection(D,eigvec[:,0]) #projecting data on to first eigen direction
a2=projection(D,eigvec[:,1]) #projecting data on to second eigen direction
projc=np.dot(D,eigvec) #projection of whole data

#scatter plot: first eigen direction
plt.scatter(x,y,marker='x',c='blue')
plt.title("projected values onto the first eigen directions")
plt.xlabel("x1")
plt.ylabel("x2")
plt.quiver(*origin,*eigvec[:,0],color="red",scale=7)
plt.quiver(*origin,*eigvec[:,1],color="red",scale=3)
plt.scatter(a1[:,0],a1[:,1],marker="x",color="magenta")
plt.axis("equal")
plt.show()

#scatter plot: second eigen direction
plt.scatter(x,y,marker='x',c='blue')
plt.title("projected values onto the second eigen directions")
plt.xlabel("x1")
plt.ylabel("x2")
plt.quiver(*origin,*eigvec[:,0],color="red",scale=7)
plt.quiver(*origin,*eigvec[:,1],color="red",scale=3)
plt.scatter(a2[:,0],a2[:,1],marker="x",color="magenta")
plt.axis("equal")
plt.show()

#(d)
print("\n(d)")
#Reconstructing the data
D_rec=np.dot(projc,eigvec.T)
print("Reconstruction error = ",((D-D_rec)**2).sum()/len(D)) #calculating mean square error

#QUES3
print("\n*******QUESTION-3*******")
#(a)
print("\n(a)")

#PCA to reduce dimension from 7 to 2
pca = PCA(n_components = 2)
df_red=pca.fit_transform(df_stand)

eval,evec=np.linalg.eig(np.cov(df_stand.T)) #eigenvalues and eigenvectors of standardised data
eval=sorted(eval,reverse=True) 

print("Variance along the 2 directions = ",np.var(df_stand.T[0]),np.var(df_stand.T[1]))
print("Eigenvalues of the 2 directions=",eval[0],eval[1])
plt.scatter(df_red.T[0],df_red.T[1],marker="x",c="blue") #plotting reduced dimesion data
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Plot after reducing dimension")
plt.show()

#b)
print("\n(b)")
x=[1,2,3,4,5,6,7]
plt.bar(x,eval,color="green") #plotting eigen values
plt.title("Eigen values in descending order")
plt.ylabel("Eigen values")
plt.show()

#c)
print("\n(c)")
rmse=[]
#calculating rmse for different values of l
for i in x:
    pca=PCA(n_components=i)
    d_rd=pca.fit_transform(df_stand) #reduced dimension data
    d_rec=pca.inverse_transform(d_rd) #reconstructed data
    rmse.append((((df_stand.values-d_rec)**2).sum()/len(d_rec))**.5)
    
plt.bar(x,rmse,color="green")
plt.ylabel("RMSE")
plt.xlabel("Reduced dimension: 'l'")
plt.title("Reconstruction Error")
plt.show()
