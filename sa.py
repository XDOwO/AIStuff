from random import sample, shuffle, uniform
import numpy as np
from math import exp
import matplotlib.pyplot as plt
N=100
city_distance=np.random.randint(1,10,[N,N])
MAXT=1000
MINT=0.01
RATE=0.99
K=50
SAMPLE_LIST=[i for i in range(1,N)]
def sa(start,solution=[]):
    def dis(solution):
        return sum([city_distance[solution[i]][solution[i-1]] for i in range(1,len(solution))])
    def change(solution):
        pos=sample(SAMPLE_LIST,2)
        solution[pos[0]],solution[pos[1]]=solution[pos[1]],solution[pos[0]]
    if start!=-1:
        solution=[i for i in range(N)]
        solution.remove(start)
        shuffle(solution)
        solution=[start]+solution+[start]
    else:
        pass
    total_distance=dis(solution)
    t=MAXT
    dislist=[total_distance]
    print("i:",solution)
    print("i dis:",total_distance)
    while t>MINT:
        for k in range(K):
            solution2=solution.copy()
            change(solution2)
            new_distance=dis(solution2)
            diff = new_distance - total_distance
            if diff<=0:
                solution=solution2
                total_distance=new_distance
            else:
                prob=exp(-diff/t)
                randnum=uniform(0,1)
                if randnum<prob:
                    solution=solution2
                    total_distance=new_distance
                else:
                    pass
        t*=RATE
        dislist.append(total_distance)
    # print(solution)
    # print(total_distance)
    # plt.figure(figsize = (15,8))
    # plt.xlabel("Iteration",fontsize = 15)
    # plt.ylabel("Distance",fontsize = 15)

    # plt.plot(dislist,linewidth = 2.5, label = "Everytime smallest distance", color = 'r')
    # plt.legend()
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()
    return solution
# for i in range(10):
#     solution=sa(1)
    # sa(-1,solution)