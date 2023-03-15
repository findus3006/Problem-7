import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def NFD(unow,uold,h,dt):
    dx = np.diff(unow,2,axis=0)[:,1:-1]
    dy = np.diff(unow,2,axis=1)[1:-1,:]
    pnew = 2*unow[1:-1,1:-1] - uold[1:-1,1:-1] + (dt**2)*(h**-2)*(dx+dy)
    return pnew

N = 100; # Points of discretization
tmax = 10; # endtime

x = np.linspace(-0.5,0.5, N)
y = np.linspace(-0.5,0.5, N)
x, y = np.meshgrid(x, y)
h = abs(x[0,2]-x[0,1]); 

m = 0.7; # Multiple for dt
dt = m*h;

t = np.arange(0,tmax,dt) # time grid


# Initialize
initial = np.zeros([N,N])
# initial = np.cos(x*np.pi)*np.cos(np.pi*y)
initial = -0.4*np.exp(-200*(x**2+y**2))
# initial = np.sinc(2*np.pi*x)*np.sinc(2*np.pi*y)

# Naive finite differences
unow = np.zeros([len(t),len(x),len(y)])
unow[0,:,:] = initial; # Initialize
uold = initial
for i in range(len(t)-1):
    unew = NFD(unow[i,:,:],uold,h,dt);
    uold = unow[i,:,:];
    unow[i+1,1:-1,1:-1] = unew;

# fig = plt.figure();
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(x,y,initial)


# Plotting
fig = plt.figure();
ax = fig.add_subplot(projection='3d')

def update(i):
    ax.clear()
    ax.plot_surface(x,y,unow[i,:,:])
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5,0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$u(r,t)$')
    ax.text(0.35,0.8,0.7,f't = {round(t[i],2)}',fontsize='12')
    #ax.legend(['FDTD','Analytical'],loc='upper left')
    ax.grid()

ani = animation.FuncAnimation(fig, update,frames = np.arange(0, len(t), 10), interval = 33.3)
plt.show()

# f = r"C:\Users\Stefan\Desktop\Uni\Master\2.Semester\Computational Physics\Python_scripts\Problem-7\StoneInPond.gif"
# writergif = animation.PillowWriter(fps=10) 
# ani.save(f, writer=writergif) 