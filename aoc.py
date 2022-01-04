from matplotlib import pyplot as plt

import numpy as np

def tabu_iterate(citynum,antnum,path_matrix,length,rho, tabu_matrix,Q):
    '''
    update pheromone matrix 信息素更新,返回新的tabu矩阵
    '''
    delta_tabu = np.zeros((citynum,citynum))
    for i in range(antnum):
        for j in range(citynum-1):
            delta_tabu[path_matrix[i][j]][path_matrix[i][j+1]] += Q/length[i]
            delta_tabu[path_matrix[i][j+1]][path_matrix[i][j]] += Q/length[i]#double mark双边标记
    return (1-rho)*tabu_matrix + delta_tabu    

def aco_algorithm(antnum,iterate_max, rho, alpha, beta, Q, citydata):    
    '''
    citydata为城市数据组
    antnum 蚂蚁数 
    iteratenum: 迭代次数 
    rho: 信息挥发系数
    tabu0: tabu_{ij}(0)
    '''
    citynum = len(citydata)
    iterate_num = 0 #initial iteration number迭代次数初始化
    dist = np.zeros((citynum,citynum)) # distance
    for i in range(citynum):
        for j in range(citynum):
            dist[i][j] = np.sqrt(pow((citydata[i][1]-citydata[j][1]),2)+pow((citydata[i][2]-citydata[j][2]),2))
    length_avg = np.zeros(iterate_max) #average length of every iteration 平均距离
    length_tmpbest = np.zeros(iterate_max) #optimal path length of all iterations 所有代最佳距离
    path_best = np.zeros((iterate_max,citynum)) #optimal path  of all iterations 所有迭代的最佳路径
    tabu_matrix = np.ones((citynum,citynum)) #pheromone matrix 信息素矩阵
    tabu_matrix = tabu_matrix/100
    eta_matrix = 1/(dist+np.diag([10e1]*citynum)) #heuristic matrix  启发函数矩阵 eta 自身可见度设为-1
    path_matrix = np.zeros((antnum,citynum)).astype(int) #path of this iteration所有蚂蚁的分散路径表,表示目前m只蚂蚁分布于哪个城市
    #iteration loop 开始外层迭代循环
    while iterate_num < iterate_max :
        '''
        将 m只蚂蚁分散到n个城市，并将其存入path矩阵
        
        '''
        path_matrix[:,0] = np.random.permutation(range(0,citynum))[:antnum] # 0 to citynum-1
        #print("path_matrix",path_matrix)
        current_city_list = [0 for i in range(antnum)]
        for a in range(antnum):
            current_city_list[a] = path_matrix[a,0]
        #print("current_city_list",current_city_list)
        unvisit_set = np.empty([antnum],dtype=set)
        #record path length of this iteration 记录所有路径长度
        length = np.zeros(antnum)
        for m in range(antnum):
            unvisit_set[m] = set(range(citynum))
            unvisit_set[m].remove(current_city_list[m])
        #print("unvisit_set", unvisit_set)
        for i in range(antnum):
            #需要走完全程
            for j in range(1,citynum): #注意是从1开始
                #选择城市
                visit = current_city_list[i]
                unvisit_list = list(unvisit_set[i]) # allowed K
                #print("unvisit_list(ant) = ", unvisit_list)
                prob = np.zeros(len(unvisit_list)) #city probability list各城市概率列表，注意与set对应关系
                for k in range(len(unvisit_list)):
                    prob[k] = np.power(tabu_matrix[visit][unvisit_list[k]],alpha)*np.power(eta_matrix[visit][unvisit_list[k]],beta)
                prob = prob/sum(prob)
                #roulette probability轮盘赌选择，进行累加
                prob = prob.cumsum()
                prob -= np.random.rand() #roulette algorithm
                #print("prob=", prob)
                nextcity = unvisit_list[np.where(prob>0)[0][0]]#choose next city选择下一个城市 [0][0]是因为返回的是一个二维数组
                #选择下一个城市之后，记录长度，添加进入路线，更新当前城市
                length[i] += dist[current_city_list[i]][nextcity]
                unvisit_set[i].remove(nextcity)
                current_city_list[i] = nextcity
                #print(nextcity)
                path_matrix[i,j] = nextcity
            length[i] += dist[current_city_list[i]][path_matrix[i][0]] #back to start city最后一个城市回到原来城市
            #print(path_matrix[i])
        #一次迭代结束后
        #print("length_min:", length.min())
        length_avg[iterate_num] = length.mean() #avg update 平均值
        if iterate_num == 0 :
            length_tmpbest[iterate_num] = length.min()
            path_best[iterate_num] = path_matrix[length.argmin()].copy()
        else: # if this iteration get better value
            if length.min() < length_tmpbest[iterate_num-1]: #update min value
                length_tmpbest[iterate_num] = length.min()
                path_best[iterate_num] = path_matrix[length.argmin()].copy()
                print("update length:",length.min())
                print("iteration num:", iterate_num)
            else: # if this iteration don't get better value
                length_tmpbest[iterate_num] = length_tmpbest[iterate_num-1]
                path_best[iterate_num] = path_best[iterate_num-1].copy()
        #update pheromone matrix 信息素更新
        tabu_matrix = tabu_iterate(citynum,antnum,path_matrix,length,rho,tabu_matrix,Q)  
        iterate_num += 1 #迭代计数器+1 
    final_path = []
    for i in range(citynum):
        final_path.append(int(path_best[-1][i]))
    final_path.append(final_path[0])#回到起点
    print("final_path: ",final_path)
    print("minium_length: ",length_tmpbest[-1])
    pathplot(antnum,alpha,beta,iterate_max,final_path,citydata,length_tmpbest,length_avg,rho)
    return final_path, length_tmpbest[-1] 

def pathplot(antnum,alpha,beta,iterate_max,final_path, citydata,length_tmpbest,length_avg,rho):
    position = np.array(citydata,dtype=int)
    position.tolist()
    x_position = []
    y_position = []
    for i in range(len(final_path)):
        x_position.append(citydata[final_path[i]][1])
        y_position.append(citydata[final_path[i]][2])
    plt.figure(dpi=120,figsize=(8,6))
    plt.subplot(211)
    plt.title("rho:"+str(rho)+" alpha:"+str(alpha)+" beta:"+str(beta)+" antnum:"+str(antnum)+" iterate_num:"+ str(iterate_max))
    plt.plot(x_position,y_position,color='blue')
    for i in range(len(final_path)):
        plt.text(x_position[i],y_position[i],str(final_path[i]),fontsize=5)
    plt.subplot(212)
    plt.plot(range(len(length_avg)),length_avg,color='grey')
    plt.plot(range(len(length_avg)),length_tmpbest, color = 'red')
    plt.legend(["length_avg","length_min"])
    plt.text(len(length_avg)*0.8,length_tmpbest[-1]*1.05,"lengthmin="+str(length_tmpbest[-1]),color = 'red')
    
    plt.show()

def aco_algorithm_mmas(antnum,iterate_max, rho, alpha, beta, citydata):    
    '''
    MMAS版本
    '''
    citynum = len(citydata)
    iterate_num = 0 #迭代次数初始化
    '''生成距离矩阵dist'''
    dist = np.zeros((citynum,citynum)) # 距离矩阵
    for i in range(citynum):
        for j in range(citynum):
            dist[i][j] = np.sqrt(pow((citydata[i][1]-citydata[j][1]),2)+pow((citydata[i][2]-citydata[j][2]),2))
    length_avg = np.zeros(iterate_max) #各代平均路径
    length_tmpbest = np.zeros(iterate_max) #最优路径长度
    path_best = np.zeros((iterate_max,citynum)) #最优路径
    tabu_matrix = np.ones((citynum,citynum)) #信息素矩阵
    tabu_matrix = tabu_matrix*10 #max为任意较大值
    eta_matrix = 1/(dist+np.diag([10e1]*citynum)) #启发函数矩阵 eta 自身可见度设为-1
    path_matrix = np.zeros((antnum,citynum)).astype(int) #分散路径表,表示目前m只蚂蚁分布于哪个城市中
    #开始外层迭代循环
    while iterate_num < iterate_max :
        '''
        将 m只蚂蚁分散到n个城市，并将其存入path矩阵
        '''
        path_matrix[:,0] = np.random.permutation(range(0,citynum))[:antnum] # 0 to citynum-1
        current_city_list = [0 for i in range(antnum)]
        for a in range(antnum):
            current_city_list[a] = path_matrix[a,0]
        unvisit_set = np.empty([antnum],dtype=set)
        #pbest
        #记录所有路径长度
        length = np.zeros(antnum)
        for m in range(antnum):
            unvisit_set[m] = set(range(citynum))
            unvisit_set[m].remove(current_city_list[m])
        for i in range(antnum):
            #需要走完全程
            for j in range(1,citynum): #注意是从1开始
                #选择城市
                visit = current_city_list[i]
                unvisit_list = list(unvisit_set[i])
                #print("unvisit_list(ant) = ", unvisit_list)
                prob = np.zeros(len(unvisit_list)) #各城市概率列表，注意与set对应关系
                #print('prob_num=', len(prob))
                for k in range(len(unvisit_list)):
                    prob[k] = np.power(tabu_matrix[visit][unvisit_list[k]],alpha)*np.power(eta_matrix[visit][unvisit_list[k]],beta)
                prob = prob/sum(prob)
                #轮盘赌选择，进行累加
                prob = prob.cumsum()
                prob -= np.random.rand()
                #print("prob=", prob)
                nextcity = unvisit_list[np.where(prob>0)[0][0]]#选择下一个城市 [0][0]是因为返回的是一个二维数组
                #选择下一个城市之后，记录长度，添加进入路线，更新当前城市
                length[i] += dist[current_city_list[i]][nextcity]
                unvisit_set[i].remove(nextcity)
                current_city_list[i] = nextcity
                #print(nextcity)
                path_matrix[i,j] = nextcity
            length[i] += dist[current_city_list[i]][path_matrix[i][0]] #最后一个城市回到原来城市
        #一次迭代结束后
        length_avg[iterate_num] = length.mean() #平均值
        if iterate_num == 0 :
            length_tmpbest[iterate_num] = length.min()
            path_best[iterate_num] = path_matrix[length.argmin()].copy()
        else:
            if length.min() < length_tmpbest[iterate_num-1]: #比当前最小的还小
                length_tmpbest[iterate_num] = length.min()
                path_best[iterate_num] = path_matrix[length.argmin()].copy()
                print("更新length:",length.min())
                print("迭代次数:", iterate_num)
            else:
                length_tmpbest[iterate_num] = length_tmpbest[iterate_num-1]
                path_best[iterate_num] = path_best[iterate_num-1].copy()
        '''
        迭代结束进行tabu_max和tabu_min的更新
        '''
        #信息素更新
        #print("当前局部最小路径path_matrix[length.argmin()]",path_matrix[length.argmin()])
        #print("全局最小path_best[iterate_num]",path_best[iterate_num])
        tabu_matrix = tabu_iterate_mmas(citynum,path_matrix[length.argmin()],path_best[iterate_num],length.min(),length_tmpbest[iterate_num],rho,tabu_matrix)  
        iterate_num += 1 #迭代计数器+1 
    
    final_path = []
    for i in range(citynum):
        final_path.append(int(path_best[-1][i]))
    final_path.append(final_path[0])#回到起点
    print("final_path: ",final_path)
    print("minium_length: ",length_tmpbest[-1])
    pathplot(antnum,alpha,beta,iterate_max,final_path,citydata,length_tmpbest,length_avg,rho)
    return final_path, length_tmpbest[-1] 

def tabu_iterate_mmas(citynum,current_best,global_best,length_currentbest,length_globalbest,rho,tabu_matrix):
    '''
    信息素更新,返回新的tabu矩阵(mmas)
    先计算本轮tabu_max和tabu_min的更新
    再对本轮的tabu_ij进行更新，同时判断是否在min和max内部
    current_best是本轮最小路径数组 opt ib length_currentbest 是当前迭代下最小解长度
    global_best是总计最小解gb length_globalbest是总计最小解长度
    '''
    pbest = 0.05 
    avg = citynum/2
    tabu_max = (1/(rho))*(1/(length_globalbest))
    tabu_min = (tabu_max*(1-np.power(pbest,1/citynum))) / ((avg-1)*(np.power(pbest,1/citynum)))
    delta_tabu = np.zeros((citynum,citynum))
    for i in range(citynum-1): #update pheromone value
        if(delta_tabu[current_best[i]][current_best[i+1]]==0):
            delta_tabu[current_best[i]][current_best[i+1]] = 1/length_currentbest
            delta_tabu[current_best[i+1]][current_best[i+1]] = 1/length_currentbest
        if(delta_tabu[int(global_best[i])][int(global_best[i+1])]==0):
            delta_tabu[int(global_best[i])][int(global_best[i+1])] = 1/length_currentbest
            delta_tabu[int(global_best[i+1])][int(global_best[i])] = 1/length_currentbest
    sum_matrix = (1-rho)*tabu_matrix + delta_tabu
    for i in range(citynum): #restrict the tabu value form min to max
        for j in range(citynum):
            if sum_matrix[i][j]>tabu_max : sum_matrix[i][j] = tabu_max
            elif sum_matrix[i][j]<tabu_min : sum_matrix[i][j] = tabu_min        
    return sum_matrix