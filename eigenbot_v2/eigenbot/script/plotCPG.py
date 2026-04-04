#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

simplified_mode = 0

if simplified_mode == 0:
    omega = 40/(np.pi*2) #oscillation frequency in rad/s
    gamma = 40 #from paper
    gait_a = np.pi/18 #from paper
    gait_b = np.pi/6 #from paper
    gait_n = 4 #from paper
    mew = 1 #from paper
    #cx0_offset = np.zeros(6)
    #cy0_offset = np.zeros(6)
    #cx0_offset = np.array([np.pi/4, np.pi/4, 0, 0, -np.pi/4, -np.pi/4]) 
    cx0_offset = np.array([np.pi/16, np.pi/16, 0, 0, -np.pi/16, -np.pi/16]) 
    cy0_offset = np.array([np.pi/16, np.pi/16, np.pi/16, np.pi/16, np.pi/16, np.pi/16]) #from paper
    Ts = .0001
    Tstart = 0
    Tstop = .11
    
    N = int((Tstop-Tstart)/Ts)
    cpg_s = (6,N+2)
    cpg_x = np.zeros(cpg_s)
    cpg_y = np.zeros(cpg_s)
    for row in range(np.shape(cpg_x)[0]):
        cpg_x[row][0]= .0
        cpg_y[row][0]= .0
    ksum0 = np.zeros(6)
    lambda_cs = 0.25 #K array coupling strength
    #coupling matrix K for tripod gait on hexapod
    K_array =   np.array(
                [[0, -1, -1, 1, 1, -1],
                [-1, 0, 1, -1, -1, 1],
                [-1, 1, 0, -1, -1, 1],
                [1, -1, -1, 0, 1, -1],
                [1, -1, -1, 1, 0, -1],
                [-1, 1, 1, -1, -1, 0]])

    #plotting
    fig, ax = plt.subplots(6,1,layout = "constrained")
    t = np.arange(Tstart,Tstop+2*Ts,Ts)

                
    #generate data

    #each time step couple each leg, i and k iteration are reversed
    for k in range(N+1):
        for i in range(6):
            H = np.absolute(((cpg_x[i][k]-cx0_offset[i])/gait_a)**gait_n) + np.absolute(((cpg_y[i][k]-cy0_offset[i])/gait_b)**gait_n)
            dHdx = (gait_n/gait_a)*((cpg_x[i][k]-cx0_offset[i])/gait_a)**(gait_n-1)
            dHdy = (gait_n/gait_b)*((cpg_y[i][k]-cy0_offset[i])/gait_b)**(gait_n-1)
            
            #reset ksum0
            ksum0[i] = 0
            for r in range(np.shape(K_array)[0]):
                for c in range(np.shape(K_array)[1]):
                    if r == i:
                        #print(K_array[r][c])
                        ksum0[i] = ksum0[i] + (K_array[r][c]*(cpg_y[i][k]-cy0_offset[i]))
                        #print(ksum)
            cpg_x[i][k+1] = (gamma*(mew - H)*dHdx - omega*(dHdy))*Ts + cpg_x[i][k]
            cpg_y[i][k+1] = (gamma*(mew - H)*dHdy + omega*(dHdx) + lambda_cs*ksum0[i])*Ts + cpg_y[i][k] #place lambda ksum inside Ts
            #print("k: " + str(k) + " X: " + str(cpg_x[k+1])+ " Y: " + str(cpg_y[k+1]))
    
    for i in range(6):
        labelCPGX = "CPG_X_LEG_" + str(i)
        labelCPGY = "CPG_Y_LEG_" + str(i)
        ax[i].plot(t, cpg_x[i], label=labelCPGX)
        ax[i].plot(t, cpg_y[i], label=labelCPGY)
        ax[i].set_xlabel('Time (t)')
        ax[i].set_ylabel('amplitude')
        ax[i].legend()
        ax[i].set_title('COMPLEX MODEL LEG '+str(i)+': CPG oscillation vs Time')
    plt.show()


#simplified model:
if simplified_mode == 1:
    omega = 300/(np.pi*2) #oscillation frequency in rad/s
    alpha = 1
    beta = 1
    gait_a = np.pi/18 #from paper
    gait_b = np.pi/6 #from paper
    gait_n = 4 #from paper
    mew = 1 #from paper
    Ts = .001
    Tstart = 0
    Tstop = .5
    lambda_cs = 1
    N = int((Tstop-Tstart)/Ts)
    cpg_s = (4,N+2)
    cpg_x = np.zeros(cpg_s)
    cpg_y = np.zeros(cpg_s)
    for row in range(np.shape(cpg_x)[0]):
        cpg_x[row][0]= .01
        cpg_y[row][0]= .01
    ksum0 = np.zeros(4)
    K_array =   np.array(
                [[0, -1, -1, 1],
                [-1, 0, 1, -1],
                [-1, 1, 0, -1],
                [1, -1, -1, 0]])

    #plotting
    fig, ax = plt.subplots(4,1,layout = "constrained")
    t = np.arange(Tstart,Tstop+2*Ts,Ts)
    for k in range(N+1):
        for i in range(4):  
            r = math.sqrt(cpg_x[i][k]**2+cpg_y[i][k]**2)
            ksum0[i] = 0
            for row in range(np.shape(K_array)[0]):
                    for col in range(np.shape(K_array)[1]):
                        if row == i:
                            #print(K_array[r][c])
                            ksum0[i] = (ksum0[i] + K_array[row][col])*cpg_y[i][k]
                            #print(ksum)
            cpg_x[i][k+1] = (alpha*(mew - r)*cpg_x[i][k] - omega*(cpg_y[i][k]))*Ts + cpg_x[i][k]
            cpg_y[i][k+1] = (beta*(mew - r)*cpg_y[i][k] + omega*(cpg_x[i][k])+ lambda_cs*ksum0[i])*Ts + cpg_y[i][k] 
    
    for i in range(4):
        labelCPGX = "CPG_X_" + str(i)
        labelCPGY = "CPG_Y_" + str(i)
        ax[i].plot(t, cpg_x[i], label=labelCPGX)
        ax[i].plot(t, cpg_y[i], label=labelCPGY)
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('amplitude')
        ax[i].legend()
        ax[i].set_title('SIMPLIFIED MODEL_'+str(i)+': CPG oscillation vs Time')
    plt.show()
    




