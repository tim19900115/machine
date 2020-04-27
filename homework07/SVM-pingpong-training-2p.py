# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%load each level data seperately
import numpy as np 
np.set_printoptions(threshold=np.inf)
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from os import listdir
from os.path import isfile, join
path = 'C:\MLGame-master\games\pingpong\log'

Frame=[]
Status=[]
Ballposition=[]
ball_speed=[]    
platform_1P=[]   
platform_2P=[]


files = listdir(path)

#with open("C:\\MLGame-master\\games\\arkanoid\\log\\2019-10-20_19-21-04.pickle", "rb") as f1:
#    data_list1 = pickle.load(f1)
#path_list = ["2019-10-20_19-21-04.pickle"]
#big_data_flag =0

for file_path in files :
  file_path = join(path, file_path)
  if isfile(file_path):
    with open(file_path, "rb") as f: 
    	data_list = pickle.load(f)

for i in range(0 , len(data_list)):
        Frame.append(data_list[i].frame)
        Status.append(data_list[i].status)
        Ballposition.append(data_list[i].ball)
        ball_speed.append(data_list[i].ball_speed)
        platform_1P.append(data_list[i].platform_1P)
        platform_2P.append(data_list[i].platform_2P)




#----------------------------------------------------------------------------------------------------------------------------
import numpy as np
Plat2X=np.array(platform_2P)[:,0][:,np.newaxis]
Plat2X_next=Plat2X[1:,:]
#Plat2Y=np.array(platform_2P)[:,0][:,np.newaxis]

instruct2=(Plat2X_next-Plat2X[0:len(Plat2X_next),0][:,np.newaxis])/5

BallX=np.array(Ballposition)[:,0][:,np.newaxis]
#print(PlatX)
BallX_next=BallX[1:,:]
print(len(BallX))
VX=(BallX_next-BallX[0:len(BallX_next),0][:,np.newaxis])

BallY=np.array(Ballposition)[:,1][:,np.newaxis]
#print(PlatX)
BallY_next=BallY[1:,:]
print(len(BallX))
VY=(BallY_next-BallY[0:len(BallY_next),0][:,np.newaxis])


Ballarray=np.array(Ballposition)[:-1]
print(len(Ballarray))
#特徵X

#特徵球x、y 平板x 球vx vy
x=np.hstack((Ballarray, Plat2X[0:-1,0][:,np.newaxis],VX,VY))
print(len(Plat2X[0:-1,0][:,np.newaxis]))
print(x)
y=instruct2

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

svm=SVC(C=3,gamma=0.002,degree=1,class_weight='balanced',decision_function_shape='ovo')#gamma = 0.01, decision_function_shape = 'ovo')#gamma='auto')##
svm.fit(x_train,y_train)
#plt.scatter(Ballarray,PlatX[0:-1,0][:,np.newaxis],VX,VY)
#plt.show()
svm_bef_scaler=svm.predict(x_test)
acc_svm_bef_scaler=accuracy_score(svm_bef_scaler,y_test)
print(acc_svm_bef_scaler)
filename ="svm-2p.sav"
pickle.dump(svm,open(filename, 'wb'))
