import numpy as np
import datetime 
import json
import math
import matplotlib
import matplotlib.pyplot as plt

from midiutil import MIDIFile

def data_2bins(data_list, nbin):
    return_data = []

    nmax = max(data_list)
    nmin = min(data_list)

    total_dist = nmax - nmin
    bin_size = total_dist/nbin

    bins = []
    for i in range(nbin):
        low = nmin + (bin_size * i)
        bins.append((i, low, low+bin_size))

    for data in data_list:
        for i in range(0, len(bins)):
            if bins[i][1] <= data < bins[i][2]:
                return_data.append(i)
    
    return return_data

def absv_data(data_list):
    return_data = []
    
    for i in data_list:
        if i < 0:
            return_data.append(-i)
        else:
            return_data.append(i)

    return return_data

def rotational_angle_mod(data_list):
    return_data = []

    for d in data_list:
        l = math.ceil(d / 3)
        n = d % 3

        if l % 2 == 0:
            n = 3 - n

        return_data.append(n)

    return return_data

def make_chromatic_midi_notes(data_list, nodenum, transpose, cliplength, res):

    now = datetime.datetime.now()
    ts = cliplength/res
    steps = int(len(data_list)/res) # Sampler Steps

    track = 0
    channel = 0
    miditime = 0
    bpm = 60 # Keep this for ease of use

    mymidi = MIDIFile(1)
    mymidi.addTempo(track, miditime, bpm)

    note_duration = ts #length of beat. Fraction of whole length
    velocity = 100

    while miditime < cliplength:
        for i in range(0, len(data_list), steps):
            print(miditime)
            
            mymidi.addNote(track, channel, data_list[i] + transpose, miditime, note_duration, velocity)
            miditime += ts

    with open("MidiFiles/notes_%d_%d%d%d.mid" % (nodenum, now.second, now.minute, now.hour), "wb") as output_file:
        mymidi.writeFile(output_file)

def make_cc_midi_(data_list, nodenum, cc, cliplength, res):

    now = datetime.datetime.now()
    ts = cliplength/res
    steps = int(len(data_list)/res) # Sampler Steps

    track = 0
    channel = 0
    miditime = 0
    bpm = 60 # Keep this for ease of use

    mymidi = MIDIFile(1)
    mymidi.addTempo(track, miditime, bpm)

    while miditime < cliplength:
        for i in range(0, len(data_list), steps):
            print(miditime)
            
            mymidi.addControllerEvent(track, channel, miditime, cc, data_list[i])
            miditime += ts

    with open("MidiFiles/cc_%d_%d%d%d.mid" % (nodenum, now.second, now.minute, now.hour), "wb") as output_file:
        mymidi.writeFile(output_file)

def note_at_0(data_list, cliplength, res):

    now = datetime.datetime.now()
    ts = cliplength/res
    steps = int(len(data_list)/res)

    track = 0
    channel = 0
    miditime = 0
    bpm = 60
    note = 60

    mymidi = MIDIFile(1)
    mymidi.addTempo(track, miditime, bpm)

    note_duration = ts #length of beat. Fraction of whole length
    velocity = 100

    zero_crossings = np.where(np.diff(np.sign(data_list)))[0]

    for i in range(0, len(data_list), steps):
        for zero in zero_crossings:
            if i <= zero <= i + steps:
                mymidi.addNote(track, channel, note, miditime, note_duration, velocity)
        miditime += ts

    with open("MidiFiles/znotes_%d%d.mid" % (now.minute, now.hour), "wb") as output_file:
        mymidi.writeFile(output_file)

def topbot_modulo(data_list):
    return_data = []

    for i in range(len(data_list)):

        thresh = math.ceil(data_list[i-1])
        low = math.floor(data_list[i-1])

        while thresh % 3 != 0:
            thresh += 1
        while low % 3 != 0:
            low -= 1

        if data_list[i-1] <= thresh and data_list[i] > thresh:
            return_data.append(0)
        elif data_list[i-1] > low and data_list[i] <= low:
            return_data.append(0)
        else:
            return_data.append(1)

    return return_data

def pitch_note_at_0(data_list_0, data_list_pitch, nodenum, transpose, cliplength, res):

    reslist = data_list_pitch
    maplist = data_list_0

    if len(data_list_0) > len(data_list_pitch):
        reslist = data_list_0
        maplist = data_list_pitch

    now = datetime.datetime.now()
    ts = cliplength/res
    steps = int(len(reslist)/res)
    smallsteps = len(maplist)/res

    track = 0
    channel = 0
    miditime = 0
    bpm = 60

    mymidi = MIDIFile(1)
    mymidi.addTempo(track, miditime, bpm)

    note_duration = ts #length of beat. Fraction of whole length
    velocity = 100

    zero_crossings = np.where(np.diff(np.sign(data_list_0)))[0]

    j = 0
    for i in range(0, len(reslist), steps):
        j = math.floor(j)
        for zero in zero_crossings: 
            if i <= zero <= (i + steps):
                
                #print('i', i, zero, j)
                mymidi.addNote(track, channel, data_list_pitch[j] + transpose, miditime, note_duration, velocity)
        miditime += ts
        j += smallsteps


    with open("MidiFiles/pznotes_%d_%d%d%d.mid" % (nodenum, now.second, now.minute, now.hour), "wb") as output_file:
        mymidi.writeFile(output_file)

#The file to Open
with open("DataForMusic/8Pend1min.json", 'r') as rfile:
    data = json.load(rfile)

##Pitch Plus Zero
i = 8
nodepitch = data['node_{}'.format(i)][3] # Y height
nodepitch = data_2bins(nodepitch, 60)
nodezero = topbot_modulo(data['node_{}'.format(i)][0])

pitch_note_at_0(nodezero, nodepitch, i, 36, 60.0, 1000)


##Make CC data
"""
for i in range(8):
    nodenum = i + 1
    node = (data['node_{}'.format(nodenum)][2])
    #node = absv_data(data['node_{}'.format(nodenum)][2])
    #node = rotational_angle_mod(node)
    node = data_2bins(node, 128)
    make_cc_midi_(node, nodenum, 3, 60, 1000)"""

# MAke Notes 
'''
nodenum = 8
node = data['node_{}'.format(nodenum)][3]
#node = rotational_angle_mod(node)
node = data_2bins(node, 48)
make_chromatic_midi_notes(node, nodenum, 36, 60, 300)'''

#gives note at zero
'''
node = data['node_5'][2]
note_at_0(node, 60, 1000)'''


#Plotting
'''
node = data['node_8'][0]
node = absv_data(node)
node = rotational_angle_mod(node)
fig, ax = plt.subplots(1)
ax.plot(node)
plt.show()'''

