
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file = pd.read_csv("pima_indians_diabetes_original.csv")

#QUES1
print("\n******QUESTION-1*******")
file1 = pd.read_csv("pima_indians_diabetes_miss.csv")

Df = file1.isnull().sum()
y = Df.values[0:9]
x = ["pregs","plas","pres","skin","test","BMI","pedi","Age","class"]

plt.bar(x,y)
plt.xlabel("Attributes")
plt.ylabel("No. of missing values")
plt.show()

#QUES2
#2(a)
print("\n******QUESTION-2(a)*******")
file2a = pd.read_csv("pima_indians_diabetes_miss.csv")

df = pd.DataFrame(file2a)
totDel=0
rowNo1=[]
row=[]

#deleting tuples having equal to or more than 1/3 of attributes with missing values:
for i in range(len(df.index)) : 
    if df.iloc[i].isnull().sum() >= len(df.columns)//3 :      
        totDel+=1
        rowNo1.append(i+1)
        row.append(i)
        file2a.drop(i,inplace=True)

print("\nTotal no of tuples deleted : ",totDel)
print("\nRow nos. of deleted tuples : \n",rowNo1)

#2(b)
print("\n******QUESTION-2(b)*******")
file2b = pd.read_csv("pima_indians_diabetes_miss.csv")

totDel2=0
rowNo2=[]
classNull = pd.DataFrame(file2b["class"])

#deleting tuples with missing class attribute:
for i in range(len(classNull.index)):
    if classNull.iloc[i].isnull().sum() == 1 :
        totDel2+=1
        rowNo2.append(i+1)
        row.append(i)
        file2b.drop(i,inplace=True)

print("\nTotal no of tuples deleted : ",totDel2)
print("\nRow nos. of deleted tuples : ",rowNo2)

#QUES3
print("\n******QUESTION-3*******")
file3 = pd.read_csv("pima_indians_diabetes_miss.csv")
df = pd.DataFrame(file3)

rowDel=list(set(row))        #total no. of rows to deleted after Q2(a) and Q2(b)

for i in rowDel : 
    df=df.drop(i)

df_null = df.isnull().sum()
y = df_null.values[0:9]

print("\nAfter Deleting tuples in Q2 :-")
print("No. of missing values in pregs : ",y[0])
print("No. of missing values in plas  : ",y[1])
print("No. of missing values in pres  : ",y[2])
print("No. of missing values in skin  : ",y[3])
print("No. of missing values in test  : ",y[4])
print("No. of missing values in BMI   : ",y[5])
print("No. of missing values in pedi  : ",y[6])
print("No. of missing values in Age   : ",y[7])
print("No. of missing values in class : ",y[8])
print("Total no. of missing values    : ",sum(y))

#QUES4
#4(a)
print("\n******QUESTION-4(a)*******")
df = pd.DataFrame(file3)

for i in rowDel : 
    df=df.drop(i)

df_1=df.fillna(df.mean())       #Replacing by mean

#Part(i)
print("\n---Part(i)---")

print("\nMean, Median, Mode and Standard Deviation after filling missing values: ")
print("\nMean:\n",df_1.mean())
print("\nMedian:\n",df_1.median())
print("\nMode:\n",df_1.mode().loc[0])
print("\nStandard Deviation:\n",df_1.std())

print("\n\nMean, Median, Mode and Standard Deviation of original file: ")
print("\nMean:\n",file.mean())
print("\nMedian:\n",file.median())
print("\nMode:\n",file.mode().loc[0])
print("\nStandard Deviation:\n",file.std())

#Part(ii)
print("\n---Part(ii)---")

print("\nRMSE value for: ")
rmse=[]
for i in df.columns:
    ind=df[i][df[i].isnull()].index
    if len(ind)!=0:
        x=0
        for j in ind:
            x+=(df_1[i][j]-file[i][j])**2
                
        x/=len(ind)
        rmse.append(round(x**0.5,4))
    else:
        rmse.append(0)
    print(i,'=',rmse[-1])
plt.bar(df.columns,rmse)
plt.ylabel('RMSE')
plt.show()

#4(b)
print("\n******QUESTION-4(b)*******")

df_2=df.fillna(df.interpolate())    #Replacing by interpolation

#Part(i)
print("\n---Part(i)---")

print("\nMean, Median, Mode and Standard Deviation after filling missing values: ")
print("\nMean:\n",df_2.mean())
print("\nMedian:\n",df_2.median())
print("\nMode:\n",df_2.mode().loc[0])
print("\nStandard Deviation:\n",df_2.std())

print("\n\nMean, Median, Mode and Standard Deviation of original file: ")
print("\nMean:\n",file.mean())
print("\nMedian:\n",file.median())
print("\nMode:\n",file.mode().loc[0])
print("\nStandard Deviation:\n",file.std())

#Part(ii)
print("\n---Part(ii)---")

print("\nRMSE value for: ")
rmse=[]
for i in df.columns:
    ind=df[i][df[i].isnull()].index
    if len(ind)!=0:
        x=0
        for j in ind:
            x+=(df_2[i][j]-file[i][j])**2
                
        x/=len(ind)
        rmse.append(round(x**0.5,4))
    else:
        rmse.append(0)
    print(i,'=',rmse[-1])
plt.bar(df.columns,rmse)
plt.ylabel('RMSE')
plt.show()

#QUES5
print("\n******QUESTION-5*******")

#Part-(i)
print("\n---Part(i)---")

Age   = df_2["Age"]         #Outliers for Age
Age_Q1=np.quantile(Age,.25)
Age_Q3=np.quantile(Age,.75)
Age_IQR=Age_Q3-Age_Q1
Age_Out=[]
ind1=[]

for i in range(len(Age.index)):
    if Age.iloc[i].item()<=(Age_Q1-1.5*(Age_IQR)) or Age.iloc[i].item()>=(Age_Q3+1.5*(Age_IQR)):
        Age_Out.append(Age.iloc[i].item())
        ind1.append(i)

print("\nOutliers for Age : ",Age_Out)

BMI   = df_2["BMI"]         #Outliers for BMI
BMI_Q1=np.quantile(BMI,.25)
BMI_Q3=np.quantile(BMI,.75)
BMI_IQR=BMI_Q3-BMI_Q1
BMI_Out=[]
ind2=[]

for i in range(len(BMI.index)):
    if BMI.iloc[i].item()<=(BMI_Q1-1.5*(BMI_IQR)) or BMI.iloc[i].item()>=(BMI_Q3+1.5*(BMI_IQR)):
        BMI_Out.append(BMI.iloc[i].item())
        ind2.append(i)

print("\nOutliers for BMI : ",BMI_Out)

boxplot = file3.boxplot(column=["Age","BMI"])
plt.title("Boxplot for Age and BMI")
plt.show()

#Part-(ii)
print("\n---Part(ii)---")

for i in ind1:
    Age.iloc[i]=Age.median()         #replacing outliers with median

for i in ind2:
    BMI.iloc[i]=round(BMI.median(),2)

file3["Age"]=Age
file3["BMI"]=BMI

boxplot = file3.boxplot(column=["Age","BMI"])
plt.title("Modified Boxplot for Age and BMI")
plt.show()
