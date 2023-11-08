# %%
import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# %%
#path = '../data/reformed_data/'
def print_all_file(path):
    file_list = []
    for file in os.listdir(path):
        print(file+'\n')
        file_list.append(file)
    return file_list

# %%
def read_data(path, file):
    #read data from preprocessed data file, retrun user's records dic
    
    f=open(path+file,'r')
    
    data = [ ]#transfer data to list
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        rw = line.strip().split(',')
        data.append(rw)
    
    data_user = {}
    for item in data:
        user = item[0]
        time = datetime.strptime(item[1], '%Y-%m-%d %H:%M:%S')
        st = [time, float(item[2]), float(item[3]), str(item[4])]
        if user in data_user.keys():
            data_user[user].append(st)
        else:
            data_user[user] = []
            data_user[user].append(st)
                
    return data_user



# %%
def segment_interval_split(user_id, user_list, interval):
    #get subtrajectories by time interval/gap for someone user
    
    user_list = sorted(user_list, key=lambda l:l[0])
    
    #print(user_list)
    
    time = []
    poi = []
    for item in user_list:
        time.append(item[0].timestamp())
        poi.append(item[3])
        
    time_diff = np.diff(time)
    
    split_index = []
    for i in range(len(time_diff)):
        if time_diff[i]>interval:
            split_index.append(i+1)#split point is start of (next)new trajectory
    
    split_index = [0] + split_index 
    
    #print(split_index)
    sub_trajectory = []
    
    for i in range(1,len(split_index)):    
        sub_trajectory.append(poi[split_index[i-1]:split_index[i]])
        
    sub_trajectory.append(poi[split_index[-1]:])
    
    for i in range(len(sub_trajectory)):
        sub_trajectory[i] = [user_id] + sub_trajectory[i]
        
    return sub_trajectory
        
        
def segmenation_gap(data_user, interval): 
    #get subtrajectories by time interval/gap for all users
    
    sub_trajectories=[]
    for key in data_user.keys():
        x = segment_interval_split(key, data_user[key], interval)
        
        sub_trajectories.append(x)

    return sub_trajectories

# %%
#get subtrajectories of all users by an average/hour peroid
def segmentation_hour(data_user, hour):
    
    sub_trajectory = []
    for user in data_user.keys():
        data_user[user] = sorted(data_user[user], key=lambda l:l[0])
   
        flag=0
        ind=[0]
        for i in range(1,len(data_user[user])):
    
            if (data_user[user][i][0]-data_user[user][flag][0]).total_seconds()/3600>=hour:
                ind.append(i)
                flag=i
            else:
                continue
        
        st = []

        #sub_poi=[[user,data_user[user][0][3]]]
        sub_poi=[]

        for i in range(0,len(ind)-1):
            ll=int(ind[i])
            n = int(i+1)
            up=int(ind[n])
            st=data_user[user][ll:up] 
    
            sub = []
            for item in st: 
                sub.append(item[3])
            sub_poi.append([user]+sub)
      
        sub_trajectory.append(sub_poi)
        
    return  sub_trajectory
    
    
    

# %%
#get subtrajectories of all users by a calendar day
def segmentation_day(data_user):
    sub_trajectories=[]
    for user in data_user.keys():

        data_date={}
        for item in data_user[user]:
            if item[0].date().strftime('%y %m %d') in data_date.keys():
                data_date[item[0].date().strftime('%y %m %d')].append(item[3])

            else:
                data_date[item[0].date().strftime('%y %m %d')]=[]
                data_date[item[0].date().strftime('%y %m %d')].append(item[3])
        sub=[]
        for key in data_date.keys():
             sub.append([user]+data_date[key])
               
        sub_trajectories.append(sub)
            
    return sub_trajectories

def write_dat(path,file,subs):
    textfile = open(path+file+'.dat', "w")
    for elements in subs:
        for poi in elements:
            for i in range(len(poi)-1):    
                textfile.write(str(poi[i])+' ')
            textfile.write(str(poi[-1])+'\n')

    textfile.close()


# %%
#statistics
def statistics(subs):
    print('user numbers:',len(subs))

    length=[]
    num_tra=[]
    for sub in subs:
        length.append(len(sub))

        for tra in sub:
            num_tra.append(len(tra)-1)

    print('total number of trajectories:',sum(length),'\n',
          'average number of trajectories:', np.round(np.mean(length),2),'\n',
          'POI level',len(num_tra),'\n',
          'average number of POI in trajectories:',np.round(np.mean(num_tra),2),'\n',
          'minimum number of POI in trajectories:',min(num_tra),'\n',
          'maximum number of POI in trajectories:',max(num_tra),'\n',
          'number of trajectories which including only one POI:',len([n for n in num_tra if n==1]),np.round(len([n for n in num_tra if n==1])/sum(length),4) )







if __name__ == '__main__':
    path = './data/reformed_data/'
    file_list = print_all_file(path)

    ##choice the data
    while True:
            print("select the data name you want to segment, {}:".format(file_list))
            file = input()
        
            if str(file) in file_list:
                break
            
            else:
                print("your input is not a valid segmentation method, please select the data name you want to segment, {}:".format(file_list))
                continue

    print('the data you choised is [ {} ], reading data...'.format(file))
    df = read_data(path,file)
    print('data reading finished')



    ##choice the segmentation method from ('hour','day','gap')
    method=['hour','day','gap']
   
    while True:
        print("input the segmentation method('hour','day','gap'):")
        segmentation_method = input()
      
        if str(segmentation_method) in method:
            break
           
        else:
            print("your input is not a valid segmentation method, please input the segmentation method('hour','day','gap')")
            continue

    print('the segmentation method you choosed is [ {} ], processing...'.format(segmentation_method))

# segmentation_method == gap
    if str(segmentation_method) == 'gap':
        
        while True:
            try:
                print("input the time gap(unit:hour):  ")
                time_interval = float(input())
            except ValueError:
                 print("Sorry, I didn't understand that.")
                 continue
      
            if time_interval <= 24:
                interval = time_interval*3600
                break
           
            else:
                print("your input is not a valid number, please input a number between 0-24")
                continue
        

        subs = segmenation_gap(df, interval)
        print("segmentation is processing...")

# segmentation_method == hour
    elif str(segmentation_method) == 'hour':
        
        while True:
            try:
                print("input the hour(unit:hour):  ")
                time_interval = float(input())
            except ValueError:
                 print("Sorry, I didn't understand that.")
                 continue
      
            if time_interval <= 24:
                interval = time_interval
                break
           
            else:
                print("your input is not a valid number, please input a number between 0-24")
                continue
        

        subs = segmentation_hour(df, interval)
        print("segmentation is processing...")

# segmentation_method == day
    elif str(segmentation_method) == 'day':

        subs = segmentation_day(df)
        print("segmentation is processing...")
        time_interval = '1'

#output the segmented data     
    output_path = './data/segmented_data/'  
    file_name = file.split('_')[0]+'_'+str(segmentation_method)+'_'+str(time_interval)+'seg'
    write_dat(output_path,file_name,subs)
    print('the segmented data writing is finished.')

    print('*'*10)
    print('the statistic of {} data segmented by {}:'.format(file.split('_')[0],segmentation_method))
    statistics(subs)
    print('*'*10)
    

        
   