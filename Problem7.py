import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def FDTD(unow,uold,h,dt):
    dx = np.diff(uold,2,axis=0)[:,1:-1]
    dy = np.diff(uold,2,axis=1)[1:-1,:]
    pnew = 2*unow[1:-1,1:-1] - uold[1:-1,1:-1] + (dt**2)*(h**-2)*(dx+dy)
    return pnew

N = 50; # Points of discretization
tmax = 10; # endtime
m = 0.9; # Multiple for dt
x = np.linspace(-0.5,0.5, N)
y = np.linspace(-0.5,0.5, N)
x, y = np.meshgrid(x, y)
h = abs(x[0,2]-x[0,1]);
dt = m*h;
t = np.arange(0,tmax,dt) # time grid
initial = np.zeros([N,N])
initial = np.cos(x*np.pi)*np.cos(np.pi*y)

fig = plt.figure();
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x,y,initial)
plt.close()

# Direct FDTD approach
unow = np.zeros([len(t),len(x),len(y)])
unow[0,:,:] = initial; # Initialize
uold = initial
for i in range(len(t)-1):
    pnew = FDTD(unow[i,:,:],uold,h,dt);
    pold = unow[i,:,:];
    unow[i+1,1:-1,1:-1] = pnew;


# Plotting
fig = plt.figure();
ax = fig.add_subplot(projection='3d')

def update(i):
    ax.clear()
    ax.plot_surface(x,y,unow[i,:,:])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.text(0.35,0.8,0.9,f't = {round(t[i],2)}',fontsize='12')
    #ax.legend(['FDTD','Analytical'],loc='upper left')
    ax.grid()

ani = animation.FuncAnimation(fig, update,frames = np.arange(0, len(t), 10), interval = 33.3)
plt.show()