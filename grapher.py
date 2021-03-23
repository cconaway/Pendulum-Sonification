import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import numpy as np
import math

with open("DataForMusic/3Pend_2ndTry.json", 'r') as rfile:
    data = json.load(rfile)

def rotational_angle_mod(data_list):
    return_data = []

    for d in data_list:
        l = math.ceil(d / 3)
        n = d % 3

        if l % 2 == 0:
            n = 3 - n

        return_data.append(n)

    return return_data

def absv_data(data_list):
    return_data = []
    
    for i in data_list:
        if i < 0:
            return_data.append(-i)
        else:
            return_data.append(i)

    return return_data

'''
n = 3
c = np.zeros((n,3))

for i in range(n):    
    c[i] = cm.jet(i/(n-1))[:3]

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('N pendulum')
ax1.title.set_text('angle')
ax2.title.set_text('angular velocity')

for i in range(n):
    activenode = 'node_{}'.format(i + 1)
    ax1.plot(data['total_time'], data[activenode][0], color=c[i,:])
    ax2.plot(data['total_time'], data[activenode][1], color=c[i,:])
fig.tight_layout()'''

node3 = absv_data(data['node_3'][0])
node3 = rotational_angle_mod(node3)

fig, ax = plt.subplots(1)
ax.title.set_text('angle, node 3') 
ax.plot(data['total_time'], node3, color='darkred')

plt.show()