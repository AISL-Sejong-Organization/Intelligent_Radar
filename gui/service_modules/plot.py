import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as FA
def seperater(arr):
    # global maximum
    pre_data = arr
    amp = np.abs(pre_data)
    # amp = amp / maximum
    phs = np.angle(pre_data)
    # phs = (phs - (- pi)) / (pi - (- pi))
    # sin = np.sin(phs)
    # sin = (sin + 1) / 2
    # seperated_data = np.stack((amp.T,sin.T), axis=0)
    # seperated_data = np.expand_dims(seperated_data, axis=0)
    return amp, phs
def animate(frame):
    x = np.linspace(0, len(amp[frame]), len(amp[frame]))
    y = amp[frame]
    line.set_data(x,y)
    return line,
data = np.load('/Users/joonghocho/Radar/Intelligent_Radar_legacy/gui/service_modules/data_copy/test/can/can2.npy')
amp, phs = seperater(data)
x, y = [], []
fig, ax = plt.subplots()
line, = ax.plot(x, y)
# plt.plot(amp[2])
ani = FA(fig, animate, frames=10, interval=100, repeat=True)
ani.save(r'./animation.gif',fps=10)
plt.show()