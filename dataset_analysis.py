import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.pyplot import show,bar,xlabel,ylabel,title,xticks,boxplot
import matplotlib.pyplot as plt


#load regression dataset
california_housing = fetch_california_housing()
x,y = california_housing.data,california_housing.target

#make the columns realistic
x[:,0:1] *= 10000
x[:,4:5] *= 10
y[:] *= 100000
x = np.delete(x,[5],axis=1)
print(x[0]," ",y[0],"\n\n")

#split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train,x_test,y_train,y_test = np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)


#define methods 


def calculate_median(x,i):
    if i == 7:
        col_of_interest = y
    else:
        col_of_interest = x[:,i:i+1]
        

    sorted_col = np.sort(col_of_interest,axis=0)   
        
    ind1 = round(sorted_col.shape[0]/2-1)
    ind2 = round(sorted_col.shape[0]/2)
    
    median = (sorted_col[ind1] + sorted_col[ind2]) / 2    
  
    return int(median)



def calculate_means(x,i):
    if i == 7:
        col_of_interest = y
    else:
        col_of_interest = x[:,i:i+1]    

    catch = 0

    print("col_of_interest.shape[0] = ",col_of_interest.shape[0], "col_of_interest[0] = ", int(col_of_interest[0]))

    for v in range(col_of_interest.shape[0]):
        catch += int(col_of_interest[v])

     

    return catch / col_of_interest.shape[0]  


def calculate_variance(mean,i):

    if i == 7:
        col_of_interest = y

    else:
        col_of_interest = x[:,i:i+1]   

    squared_disatance_from_mean = []
    print("col_of_interest[0] here <<  = ",col_of_interest[0])    

    for m in range(col_of_interest.shape[0]):
        sqrdiff = (mean - col_of_interest[m])**2
        squared_disatance_from_mean.append(sqrdiff)

    return np.array(squared_disatance_from_mean).sum()/col_of_interest.shape[0]


def split_numpy(*args): #10320
    
    if len(args) == 2:
        L25 = []
        ML25 = []
        MU25 = []
        U25 = []

        for i in range(y.shape[0]):
            if i < x.shape[0]/4:
                L25.append(int(y[i]))

            elif i >= x.shape[0]/4 and i < x.shape[0]/2:
                ML25.append(int(y[i]))

            elif i >= x.shape[0]/2 and i < (x.shape[0]/4*3):
                MU25.append(int(y[i]))  

            else:
               U25.append(int(y[i]))       

        return [np.array(L25),np.array(ML25),np.array(MU25),np.array(U25)]

    else: 
        L50 = []  
        U50 = []   

        for i in range(y.shape[0]):
            if i >= x.shape[0]/2:
                U50.append(int(y[i]))

            else:
                L50.append(int(y[i]))


        return np.array(L50), np.array(U50)




#create the column headers(it doesn't come with any)
column_names = [
    "Income",  
    "HouseAge",  
    "AveRooms", 
    "AveBedrooms", 
    "Population",  
    "Latitude",  
    "Longitude",  
    "house price"
]

print("x.shape = ", x.shape,"\n")


medians = []
means = []
variance = []

for i,col in enumerate(column_names):
    #print(col," ",column_names[-1])
    if col == column_names[-1]:
        medians.append(calculate_median(y,i))
        means.append(calculate_means(y,i))
    else:    
        medians.append(calculate_median(x,i))
        means.append(calculate_means(x,i))

    variance.append(calculate_variance(means[i],i))

for m in medians:
    print(m)    

print("means -------")
"""for me in means:
    print(me)    """
print("means = ", means,"\n")

medians = np.array(medians).reshape(8,1)
medians_scientific = ["{:.2e}".format(value) for value in medians.flatten()]


print(medians_scientific)

bar(column_names,medians_scientific)

ylabel("medians")
xlabel("feature names")
title("medians of dataset")
xticks(rotation=45)
show()


#mean colwise & plot histogram

means = np.array(means)
means_scientific = ["{:.2e}".format(value) for value in means.flatten()]

bar(column_names,means_scientific)

ylabel("means")
xlabel("feature names")
title("means of dataset")
xticks(rotation=45)
show()


#variance colwise & plot
variance = np.array(variance)
variance_scientific = ["{:.2e}".format(var) for var in variance.flatten()]
print("variance_scientific = ", variance_scientific)
bar(column_names,variance_scientific)
xlabel("feature names") 
ylabel("medians")
xticks(rotation=45)
show()


#house price interquartile range
y_sorted = np.sort(y,axis=0)
L50,U50 = np.split(y_sorted,2)
L50,U50 = np.sort(L50,axis=0),np.sort(U50,axis=0)
L25,ML25 = np.split(L50,2) 
MU25,U25 = np.split(U50,2) 
boxplot([L25,ML25,MU25,U25])

xticks([1, 2, 3, 4], ['L25', 'ML25', 'MU25', 'U25'])
ylabel("Values")
title("Box-and-Whisker Plot of Quartiles")
show()










