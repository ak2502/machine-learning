import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


file = pd.read_csv("landslide_data3.csv")

#QUES1
print("\n******QUESTION-1*******")

def mean(x):
    return np.mean(x)

def median(x):
    return np.median(x)

def mode(x):
    return stats.mode(x)

def minimum(x):
    return min(x)

def maximum(x):
    return max(x)

def stdDev(x):
    return np.std(x)


#printing mean,median,mode.minimum,maximum and standard deviation for all the 7 quantities

print("\n TEMPERATURE: ")
print(" Mean = ",mean(file["temperature"]))
print(" Median = ",median(file["temperature"]))
print(" Mode = ",mode(file["temperature"])[0])
print(" Minimum = ",minimum(file["temperature"]))
print(" Maximum = ",maximum(file["temperature"]))
print(" Standard Deviation = ",stdDev(file["temperature"]))

print("\n HUMIDITY: ")
print(" Mean = ",mean(file["humidity"]))
print(" Median = ",median(file["humidity"]))
print(" Mode = ",mode(file["humidity"])[0])
print(" Minimum = ",minimum(file["humidity"]))
print(" Maximum = ",maximum(file["humidity"]))
print(" Standard Deviation = ",stdDev(file["humidity"]))

print("\n PRESSURE: ")
print(" Mean = ",mean(file["pressure"]))
print(" Median = ",median(file["pressure"]))
print(" Mode = ",mode(file["pressure"])[0])
print(" Minimum = ",minimum(file["pressure"]))
print(" Maximum = ",maximum(file["pressure"]))
print(" Standard Deviation = ",stdDev(file["pressure"]))

print("\n RAIN: ")
print(" Mean = ",mean(file["rain"]))
print(" Median = ",median(file["rain"]))
print(" Mode = ",mode(file["rain"])[0])
print(" Minimum = ",minimum(file["rain"]))
print(" Maximum = ",maximum(file["rain"]))
print(" Standard Deviation = ",stdDev(file["rain"]))

print("\n LIGHTAVGW/o0: ")
print(" Mean = ",mean(file["lightavgw/o0"]))
print(" Median = ",median(file["lightavgw/o0"]))
print(" Mode = ",mode(file["lightavgw/o0"])[0])
print(" Minimum = ",minimum(file["lightavgw/o0"]))
print(" Maximum = ",maximum(file["lightavgw/o0"]))
print(" Standard Deviation = ",stdDev(file["lightavgw/o0"]))

print("\n LIGHTMAX: ")
print(" Mean = ",mean(file["lightmax"]))
print(" Median = ",median(file["lightmax"]))
print(" Mode = ",mode(file["lightmax"])[0])
print(" Minimum = ",minimum(file["lightmax"]))
print(" Maximum = ",maximum(file["lightmax"]))
print(" Standard Deviation = ",stdDev(file["lightmax"]))

print("\n MOISTURE: ")
print(" Mean = ",mean(file["moisture"]))
print(" Median = ",median(file["moisture"]))
print(" Mode = ",mode(file["moisture"])[0])
print(" Minimum = ",minimum(file["moisture"]))
print(" Maximum = ",maximum(file["moisture"]))
print(" Standard Deviation = ",stdDev(file["moisture"]))


#QUES2(a)
print("\n\n******QUESTION-2(a)*******")

X = file["rain"]

#plotting scatter plot for rain with all the other 6 quantities
plt.scatter(X,file["temperature"])
plt.xlabel("Rain")
plt.ylabel("Temperature")
plt.title("Plot between Rain and Temperature")
plt.show()

plt.scatter(X,file["humidity"])
plt.xlabel("Rain")
plt.ylabel("Humidity")
plt.title("Plot between Rain and Humidity")
plt.show()

plt.scatter(X,file["pressure"])
plt.xlabel("Rain")
plt.ylabel("Pressure")
plt.title("Plot between Rain and Pressure")
plt.show()

plt.scatter(X,file["lightavgw/o0"])
plt.xlabel("Rain")
plt.ylabel("Lightavgw/o0")
plt.title("Plot between Rain and Lightavgw/o0")
plt.show()

plt.scatter(X,file["lightmax"])
plt.xlabel("Rain")
plt.ylabel("Lightmax")
plt.title("Plot between Rain and Lightmax")
plt.show()

plt.scatter(X,file["moisture"])
plt.xlabel("Rain")
plt.ylabel("Moisture")
plt.title("Plot between Rain and Moisture")
plt.show()


#QUES2(b)
print("\n\n******QUESTION-2(b)*******")

X = file["temperature"]

#plotting scatter plot for temperature with all the other 6 quantities
plt.scatter(X,file["humidity"])
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Plot between Temperature and Humidity")
plt.show()

plt.scatter(X,file["pressure"])
plt.xlabel("Temperature")
plt.ylabel("Pressure")
plt.title("Plot between Temperature and Pressure")
plt.show()

plt.scatter(X,file["rain"])
plt.xlabel("Temperature")
plt.ylabel("Rain")
plt.title("Plot between Temperature and Rain")
plt.show()

plt.scatter(X,file["lightavgw/o0"])
plt.xlabel("Temperature")
plt.ylabel("Lightavgw/o0")
plt.title("Plot between Temperature and Lightavgw/o0")
plt.show()

plt.scatter(X,file["lightmax"])
plt.xlabel("Temperature")
plt.ylabel("Lightmax")
plt.title("Plot between Temperature and Lightmax")
plt.show()

plt.scatter(X,file["moisture"])
plt.xlabel("Temperature")
plt.ylabel("Moisture")
plt.title("Plot between Temperature and Moisture")
plt.show()


#QUES3(a)
print("\n\n******QUESTION-3(a)*******")

def corCoef(x,y):
    return np.corrcoef(x,y)

x = file["rain"]

#printing correlation coefficient between rain and other 6 quantities
print("\n Correlation Coefficient between Rain and: \n")
print(" Temperature  = %.8f" %corCoef(x,file["temperature"])[0][1])
print(" Humidity     = %.8f" %corCoef(x,file["humidity"])[0][1])
print(" Pressure     = %.8f" %corCoef(x,file["pressure"])[0][1])
print(" Lightavgw/o0 = %.8f" %corCoef(x,file["lightavgw/o0"])[0][1])
print(" Lightmax     = %.8f" %corCoef(x,file["lightmax"])[0][1])
print(" Moisture     = %.8f" %corCoef(x,file["moisture"])[0][1])

#QUES3(b)
print("\n\n******QUESTION-3(b)*******")

def corCoef(x,y):
    return np.corrcoef(x,y)

x = file["temperature"]

#printing correlation coefficient between temperature and other 6 quantities
print("\n Correlation Coefficient between Temperature and: \n")
print(" Rain         = %.8f" %corCoef(x,file["rain"])[0][1])
print(" Humidity     = %.8f" %corCoef(x,file["humidity"])[0][1])
print(" Pressure     = %.8f" %corCoef(x,file["pressure"])[0][1])
print(" Lightavgw/o0 = %.8f" %corCoef(x,file["lightavgw/o0"])[0][1])
print(" Lightmax     = %.8f" %corCoef(x,file["lightmax"])[0][1])
print(" Moisture     = %.8f" %corCoef(x,file["moisture"])[0][1])


#QUES4
print("\n\n******QUESTION-4*******")

rain = file["rain"]
moisture = file["moisture"]

#Plotting histogram for rain
rain.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain")
plt.show()

#Plotting histogram for moisture
moisture.hist()
plt.xlabel("Moisture(in %)")
plt.ylabel("Frequency")
plt.title("Histogram for Moisture")
plt.show()


#QUES5
print("\n\n******QUESTION-5*******")

df = pd.DataFrame(file)

#grouping the data w.r.t. stationid's
t6  = df.groupby("stationid").get_group('t6') 
t7  = df.groupby("stationid").get_group('t7') 
t8  = df.groupby("stationid").get_group('t8') 
t9  = df.groupby("stationid").get_group('t9') 
t10 = df.groupby("stationid").get_group('t10')
t11 = df.groupby("stationid").get_group('t11') 
t12 = df.groupby("stationid").get_group('t12') 
t13 = df.groupby("stationid").get_group('t13') 
t14 = df.groupby("stationid").get_group('t14') 
t15 = df.groupby("stationid").get_group('t15') 

#recording data for rain in variables r6,r7,r8,r9....,r15
r6  = t6["rain"]
r7  = t7["rain"]
r8  = t8["rain"]
r9  = t9["rain"]
r10 = t10["rain"]
r11 = t11["rain"]
r12 = t12["rain"]
r13 = t13["rain"]
r14 = t14["rain"]
r15 = t15["rain"]

#plotting histogram of rain for station t6
r6.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t6")
plt.show()

#plotting histogram of rain for station t7
r7.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t7")
plt.show()

#plotting histogram of rain for station t8
r8.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t8")
plt.show()

#plotting histogram of rain for station t9
r9.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t9")
plt.show()

#plotting histogram of rain for station t10
r10.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t10")
plt.show()

#plotting histogram of rain for station t11
r11.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t11")
plt.show()

#plotting histogram of rain for station t12
r12.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t12")
plt.show()

#plotting histogram of rain for station t13
r13.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t13")
plt.show()

#plotting histogram of rain for station t14
r14.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t14")
plt.show()

#plotting histogram of rain for station t15
r15.hist()
plt.xlabel("Rain(in mm)")
plt.ylabel("Frequency")
plt.title("Histogram for Rain in station t15")
plt.show()


#QUES6
print("\n\n******QUESTION-6*******")

#plotting boxplot of rain(original scale)
df.boxplot(column ='rain')
plt.title("BoxPlot for Rain")
plt.ylabel("Rain(in mm)")
plt.show()

#plotting boxplot of rain(for Y in range(0-5000))[for proper observation] 
df.boxplot(column ='rain')
plt.title("BoxPlot for Rain")
plt.ylabel("Rain(in mm)")
plt.ylim(0,5000)
plt.show()

#plotting boxplot of rain(for Y in range(0-50)) [for proper observation] 
df.boxplot(column ='rain')
plt.title("BoxPlot for Rain")
plt.ylabel("Rain(in mm)")
plt.ylim(0,50)
plt.show()

#plotting boxplot of moisture
df.boxplot(column ='moisture')
plt.ylabel("Moisture(in %)")
plt.title("BoxPlot for Moisture")
plt.show()
