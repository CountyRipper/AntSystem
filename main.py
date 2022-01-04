from aoc import aco_algorithm, aco_algorithm_mmas
import getdata
import time

if __name__ == "__main__":
    citydata = getdata.read_tsp("berlin52.tsp")
    '''antnum,iterate_max, rho, alpha, beta, Q, tabu0,citydata'''
    start = time.time()
    final_path, bestlength = aco_algorithm(30,500,0.5,0.5,5,1,citydata)
    end = time.time()
    print("time = ",end-start) 
    # start = time.time()
    # final_path1, bestlength1 = aco_algorithm_mmas(30,500,0.02,1,2,citydata)
    # end = time.time()
    # print("time = ",end-start)
