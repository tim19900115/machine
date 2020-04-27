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


path_list = [ 
    '001.pickle',
    '002.pickle',
    '003.pickle',
    '004.pickle',
    '005.pickle',
    '006.pickle',
    ]
Frame=[]
Status=[]
Ballposition=[]
PlatformPosition=[]
Bricks=[]
filenamelist = []


#with open("C:\\MLGame-master\\games\\arkanoid\\log\\2019-10-20_19-21-04.pickle", "rb") as f1:
#    data_list1 = pickle.load(f1)
#path_list = ["2019-10-20_19-21-04.pickle"]
#big_data_flag =0

for file_path in path_list :
    with open(file_path, "rb") as f: 
    	data_list = pickle.load(f)

for i in range(0 , len(data_list)):
        Frame.append(data_list[i].frame)
        Status.append(data_list[i].status)
        Ballposition.append(data_list[i].ball)
        PlatformPosition.append(data_list[i].platform)
        Bricks.append(data_list[i].bricks)

#----------------------------------------------------------------------------------------------------------------------------
import numpy as np
PlatX=np.array(PlatformPosition)[:,0][:,np.newaxis]
PlatX_next=PlatX[1:,:]
PlatY=np.array(PlatformPosition)[:,0][:,np.newaxis]

instruct=(PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5

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
x=np.hstack((Ballarray, PlatX[0:-1,0][:,np.newaxis],VX,VY))
print(len(PlatX[0:-1,0][:,np.newaxis]))
print(x)
y=instruct


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.02, random_state=999)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

yknn_bef_scaler=knn.predict(x_test)
acc_knn_bef_scaler=accuracy_score(yknn_bef_scaler,y_test)
print(acc_knn_bef_scaler)
filename ="KNN-W2.sav"
pickle.dump(knn,open(filename, 'wb'))
