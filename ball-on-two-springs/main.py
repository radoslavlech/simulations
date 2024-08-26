import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anm
from matplotlib import gridspec as gsp
from scipy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from DSP import *

#parameters of the air
rho_a = 1.206                                   #[kg/m^3] density of air
eta = 1.802e-5                                 #[kg.m/s] dynamical viscosity of air
Cv = 0.5                                        #coefficient for air drag V^2
g = np.array([0,-9.81])                         #[m/s^2] gravity

#parameters of the ruber bands
L = 1                        #length of a free-hanging rubber band
D = 0.6
k1 = 30
k2 = 30

#parameters of the ball
rho_b = 2000                          #[kg/m^3] density of ball
R = 0.03                                       #[m] radius of the ball
S = np.pi * R ** 2                              #[m^2]cross section of the ball
M = np.pi * 4*R**3 / 3 * rho_b                  #[kg] mass of the ball
x0 = -1
y0 = -3
r0 = np.array([x0,y0])                          #starting position of the ball
v0x = 0
v0y = 0
v0 = np.array([v0x,v0y])

equil = np.array([D/2, -np.sqrt(L**2-(D/2)**2)])

Y0 = np.hstack((r0,v0))


av2_switch = True
stokes_switch = True
gravity_swith = True
F_p_switch = False        #Turning on and off the forced vibration

spring = True               #determins if these are springs or just rubber bands

conststokes = -6*np.pi*eta*R/M #trick with M
def aStokes(Y):
    return conststokes*Y[2:]

cv2 = -Cv*S*rho_a / M
def av2(Y):
    return cv2*norm(Y[2:4])*Y[2:4]

#acceleration due to first band
def F1(Y):
    # magnitude of the force times unit vector (unit vector is directed from the current postion toward the node)
    if norm(Y[:2]) >L:
        return -(k1*(np.linalg.norm(Y[:2])-L)*Y[:2]/np.linalg.norm(Y[:2]))/M
    else:
        return -spring*(k1*(np.linalg.norm(Y[:2])-L)*Y[:2]/np.linalg.norm(Y[:2]))/M

def F2(Y):
    L2 = -Y[:2]+np.hstack((D,0))
    if norm(L2)>L:
        return (k2*(np.linalg.norm(L2)-L)*L2/np.linalg.norm(L2))/M
    else:
        return spring*(k2*(np.linalg.norm(L2)-L)*L2/np.linalg.norm(L2))/M


def F_periodic(t,Y):
    return 200*np.sin(6*2*np.pi*t)/M*np.array([1,0])



def eom(t,Y): #two dimensional case equation of motion
    return np.hstack((Y[2:],gravity_swith*g+av2_switch*av2(Y)+F1(Y)+F2(Y)+stokes_switch*aStokes(Y)+F_p_switch*F_periodic(t,Y)))


t0 = 0
te = 20                        #times of measurements
Nt = 1000                               #number of desired time points
dt_max = 0.01                          #maximum permissible length of time step VERY IMPORTANT to accurate result and more timestamps



sol1 = solve_ivp(eom, (t0,te), Y0, args=(), max_step=dt_max)
t = sol1.t
Y = sol1.y


#Plot of energies
Ek = []
Ep = []
Ediss = [0,]

E_stokes = 0
E_av2 = 0

for i in range(len(t)):
    spring1 = spring
    spring2 = spring
    Ek.append(0.5*M*np.linalg.norm(Y[2:4,i])**2)
    #potential energy of springs and of gravity
    deltax1 = norm(Y[:2,i]) - L
    deltax2 = np.linalg.norm(Y[:2,i] - np.hstack((D, 0))) - L
    if deltax1>0:
        spring1 = True
    elif deltax2>0:
        spring2 = True
    Ep.append(gravity_swith*M * 9.81 * np.abs(-10 - Y[1, i]) + spring1 * 0.5 * k1 * deltax1 **2 + spring2 * 0.5 * k2 * deltax2**2)

    try:
        E_av2 += av2_switch* M * np.linalg.norm(av2(Y[:, i])) * np.linalg.norm(Y[:2, i + 1] - Y[:2, i])
        E_stokes += stokes_switch* M *np.linalg.norm(aStokes(Y[:,i]))*np.linalg.norm(Y[:2,i+1]-Y[:2,i])
        Ediss.append(E_av2+E_stokes)
    except:
        None
Ek = np.array(Ek)
Ep = np.array(Ep)
Ediss = np.array(Ediss)
Etot = Ek+Ep+Ediss



#Signal analysis part:
t0_sam = 5
te_sam = 15
sam_start_idx = int(t0_sam/(te-t0)*len(t))
sam_stop_idx = int(te_sam/(te-t0)*len(t))
time_sample = t[sam_start_idx:sam_stop_idx]
window = Hanning(time_sample,t0_sam,te_sam)
signal = Y[0,sam_start_idx:sam_stop_idx]
sig_windowed = []
for k, second in enumerate(time_sample):
    sig_windowed.append(signal[k]*window[k])
sig_windowed = np.array(sig_windowed)

#Fast Fourier transform
N = len(sig_windowed)
omega = fftfreq(N,dt_max)[:N//2]
fourier = fft(sig_windowed)
fourier_unwind = fft(signal)




frame_n = len(t)
def update_plot(nt):
    ball_pos.set_data(Y[0,nt],Y[1,nt])
    ball_traj.set_data(Y[0,0:nt],Y[1,0:nt])
    spring1.set_data(np.linspace(0,Y[0,nt]),np.linspace(0,Y[1,nt]))
    spring2.set_data(np.linspace(Y[0,nt],D),np.linspace(Y[1,nt],0))
    x.set_data(t[0:nt],Y[0,0:nt]-equil[0])
    y.set_data(t[0:nt],Y[1,0:nt]-equil[1])
    kin_energy.set_data(t[0:nt],Ek[0:nt])
    pot_energy.set_data(t[0:nt],Ep[0:nt])
    diss_energy.set_data(t[0:nt],Ediss[0:nt])
    tot_energy.set_data(t[0:nt],Etot[0:nt])
    return ball_traj, ball_pos, spring1, spring2, x, y, kin_energy,pot_energy, tot_energy,

fig = plt.figure(figsize=(16,9), dpi=120, facecolor=(0.8,0.8,0.8))
gs = gsp.GridSpec(2,2)

ax0 = fig.add_subplot(gs[:,0], facecolor = (0.9,0.9,0.9))
ax0.hlines(0,-3,3,color = 'k', lw = 2)
ax0.plot(0,0, marker=".", markersize=10,color='k')
ax0.plot(D,0, marker=".", markersize=10,color='k')
tyt = f'Trajectory y(x)\nM = {M:.3} kg, time of the simulation: {te-t0} s, v0 = {v0} m/s'
ax0.set_ylabel('y[m]')
ax0.set_xlabel('x[m]')
ax0.set_title(tyt)

spring1, = ax0.plot([],[],linestyle='--',color='k',lw=1)
spring2, =  ax0.plot([],[],linestyle='--',color='k',lw=1)
ball_pos, = ax0.plot([],[], marker=".", markersize=R*500,color='k', label='starting point')
ball_traj, = ax0.plot([],[],'--r',label='y(x)',lw=0.5)
plt.axis('equal')
plt.xlim(-3,3)
plt.ylim(-5,5)

ax1 = fig.add_subplot(gs[0,1], facecolor = (0.9,0.9,0.9))
x, = ax1.plot([],[],'-b',label='x(t)')
y, = ax1.plot([],[],'-g',label='y(t)')
ax1.legend()
tyt1 = f'plots of x(t) and y(t)'
plt.xlim(0,20)
plt.ylim(-3,3)
ax1.set_title(tyt1)


ax2 = fig.add_subplot(gs[1,1], facecolor = (0.9,0.9,0.9))
kin_energy, = ax2.plot([],[],'-g',label='Kinetic [J]',lw=1)
pot_energy, = ax2.plot([],[],'-b',label='Potential energy [J]',lw=1)
diss_energy, = ax2.plot([],[],'-r',label='Dissipated [J]',lw=2)
tot_energy, = ax2.plot([],[],'-k',label='Total  energy [J]',lw=1)
ax2.legend()
plt.xlim(0,20)
plt.ylim(-1,450)
tyt2 = f'Examination of the conservation of energy'
ax2.set_title(tyt2)


traj_anim = anm.FuncAnimation(fig, update_plot, frames=frame_n, interval=20,repeat=False,blit=True )
plt.show()

fig2 = plt.figure(figsize=(16,9), dpi=120, facecolor=(0.8,0.8,0.8))
gs = gsp.GridSpec(2,2)
ax3 = fig2.add_subplot(gs[0,1], facecolor = (0.9,0.9,0.9))
ax3.plot(time_sample,window,'-r',label='Hanning window (t)',lw=1)
ax3.plot(time_sample,sig_windowed,'-b',label='Signal multiplied by window',lw=1)
ax3.legend()

ax4 = fig2.add_subplot(gs[:,0], facecolor = (0.9,0.9,0.9))
ax4.plot(omega,2/N*np.abs(fourier[:N//2]),'-b',label='FFT of windowed signal',lw=1)
ax4.plot(omega,1/N*np.abs(fourier_unwind[:N//2]),'--r',label='FFT of original signal',lw=0.6)
plt.xlim(0,30)
ax4.legend()

ax5 = fig2.add_subplot(gs[1,1], facecolor = (0.9,0.9,0.9))
ax5.plot(time_sample,signal,'-b',label='original signal',lw=1)


plt.show()








