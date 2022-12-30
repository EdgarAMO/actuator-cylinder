# -*- coding: utf-8 -*-


# -----------------------------------------------------------------------------
#
# importation of libraries
#
# -----------------------------------------------------------------------------
from scipy.interpolate import interp1d      # one-dimensional interpolation
from scipy.interpolate import interp2d      # two-dimensional interpolation
from scipy.integrate import simps           # simpson's rule for integration
from scipy.integrate import quad            # gaussian quadrature

import numpy as np                          # array manipulation

from math import sin                        # sine operation on scalars
from math import cos                        # cosine operation on scalars
from math import acos                       # acosine operation on scalars
from math import exp                        # exponential on scalars
from math import sqrt                       # square root on scalars
from math import acos                       # acos on scalars

from numba import jit                       # just-in-time compilation

from matplotlib.patches import Circle       # geometrical shapes
import matplotlib.pyplot as plt             # standard plot library

#------------------------------------------------------------------------------
#
# external jitted functions
#
#------------------------------------------------------------------------------
@jit(nopython=True)
def fillMatrices(cx, cy, x, y, t, N, DT):
    """ fills Cx and Cy influence matrices """

    NS = 100        # number of subdivisions in each sector
    DP = DT / NS    # subdivision width
    p = 0.00        # subdivision angle

    # loop through control points:
    for j in range(N):

        # loop through sectors:
        for i in range(N):
            cx[j, i] = 0.00     # set each element as an accumulator
            cy[j, i] = 0.00     # set each element as an accumulator

            # loop through subdivisions:
            for k in range(NS + 1):

                # subdivision angle:
                p = t[i] - (1.0 / 2.0) * DT + k * DP

                # numerator and denominator for cx:
                above =  -(x[j] + sin(p)) * sin(p) + (y[j] - cos(p)) * cos(p)
                below = (x[j] + sin(p))**2 + (y[j] - cos(p))**2
                cx[j, i] += (above / below) * DP * (1 / (-2.0 * np.pi))

                # numerator for cy:
                above =  -(x[j] + sin(p)) * cos(p) - (y[j] - cos(p)) * sin(p)
                cy[j, i] += (above / below) * DP * (1 / (-2.0 * np.pi))


@jit(nopython=True)
def getWx(x, y, phi, qn, wxgrid):
    """ returns a flattened array of wx due to the current turbine """
    
    # number of flattened grid points
    NGP = x.size       

    for j in range(NGP):
        # x & y are arrays!
        # up: numerator of quotient
        # dw: denominator of quotient
        # integrand: array of size NS
        # f: array to be integrated
        # difference or step
        # number of intervals

        up = -(x[j]+np.sin(phi))*np.sin(phi) + (y[j]-np.cos(phi))*np.cos(phi)

        dw = (x[j] + np.sin(phi)) ** 2 + (y[j] - np.cos(phi)) ** 2

        f = qn * (up / dw)

        h = np.diff(phi)[0]  

        n = f.size - 1  

        s = (h/3) * (f[0] + 2*np.sum(f[2:n:2]) + 4*np.sum(f[1:n:2]) + f[n])

        wxgrid[j] = s
        
    wxgrid = -(1 / (2 * np.pi)) * wxgrid
    
    return wxgrid


@jit(nopython=True)
def getWy(x, y, phi, qn, wygrid):
    """ returns a flattened array of wx due to the current turbine """
    
    # number of flattened grid points
    NGP = x.size        
    
    for j in range(NGP):
        # x & y are arrays!
        # up: numerator of quotient 
        # dw: denominator of quotient
        # integrand: array of size NS
        # f: array to be integrated
        # difference or step
        # number of intervals

        up = -(x[j]+np.sin(phi))*np.cos(phi) - (y[j]-np.cos(phi))*np.sin(phi)

        dw = (x[j] + np.sin(phi)) ** 2 + (y[j] - np.cos(phi)) ** 2

        f = qn * (up / dw) 

        h = np.diff(phi)[0]

        n = f.size - 1 

        s = (h/3) * (f[0] + 2*np.sum(f[2:n:2]) + 4*np.sum(f[1:n:2]) + f[n])

        wygrid[j] = s
        
    wygrid = -(1 / (2 * np.pi)) * wygrid
    
    return wygrid


# wake influence:
def getWw(x, y, qnFun, wwgrid):

    # x & y are flattened arrays
    # qnFun is a continuous functions qn(t)

    for j in range(x.size):

        shadowed = (y[j] > -0.99) and (y[j] < 0.99)
        outside = sqrt(x[j] ** 2 + y[j] ** 2) > 1.0
        downwind = x[j] > 0.
        
        # downwind:
        if shadowed and outside and downwind:
            value = - qnFun(acos(y[j])) + qnFun(2*PI - acos(y[j]))

        # inside:
        elif shadowed and not outside:
            value = - qnFun(acos(y[j]))

        # outside:
        else:
            value = 0

        # fill wwgrid:
        wwgrid[j] = value

    return wwgrid
    

# -----------------------------------------------------------------------------
#
# geometrical and operational parameters
#
# -----------------------------------------------------------------------------
PI = np.pi          # pi constant
N = 36              # number of control points
F = 1.01            # outward offset factor for the control points
E = 1e-2            # relative tolerance for convergence
K = 0.25            # under-relaxation factor
DT = 0.00           # slice width in radians

TSR0 = 0.0          # initial tip-speed ratio
TSRF = 0.0          # final tip-speed ratio
NTSR = 0            # number of tip-speed ratios
RPM = 0.00          # revolutions per minute
RAD = 0.00          # turbine's radius in meters
BLA = 0             # number of blades
CHR = 0.00          # blade's chord
SEC = 0             # wing section number

SOL = 0.00          # turbine's solidity
REG = 0.00          # global reynolds number
NU = 1.55e-5         # kinematic viscosity (default is air)


# -----------------------------------------------------------------------------
#
# load input data and unpack it
#
# -----------------------------------------------------------------------------
pack = np.loadtxt('settings.csv', delimiter=',', skiprows=1, unpack=True)

TSR, RPM, RAD, BLA, CHR, SEC, BPA = pack

# solidity, global Re and angular step in radians
SOL = (BLA * CHR) / (2 * RAD)
REG = (RPM * RAD * CHR * PI) / (30 * NU)
DT = 2 * PI / N

# blade pitch angle converted to radians
BPA *= PI / 180


# -----------------------------------------------------------------------------
#
# get cl and cd interpolation functions
#
# -----------------------------------------------------------------------------
if SEC == 1:
    clfile, cdfile = 'naca0012cl.csv', 'naca0012cd.csv'
elif SEC == 2:
    clfile, cdfile = 'naca0015cl.csv', 'naca0015cd.csv'
elif SEC == 3:
    clfile, cdfile = 'naca0018cl.csv', 'naca0018cd.csv'
elif SEC == 4:
    clfile, cdfile = 'naca0021cl.csv', 'naca0021cd.csv'
elif SEC == 5:
    clfile, cdfile = 'du06w200cl.csv', 'du06w200cd.csv'
else:
    raise Exception('Airfoil input error')

if SEC != 5:
    aaStr = 'nacaaa.csv'
    reStr = 'nacare.csv'
else:
    aaStr = 'du06w200aa.csv'
    reStr = 'du06w200re.csv'

# load arrays of coefficients:
CL = np.loadtxt('airfoils/'+clfile, delimiter=',')
CD = np.loadtxt('airfoils/'+cdfile, delimiter=',')

# angle of attack and reynolds tables:
AA = np.loadtxt('airfoils/{}'.format(aaStr), unpack=True)
RE = np.loadtxt('airfoils/{}'.format(reStr), unpack=True)

# create functions for lift and drag coefficients:
fCL = interp2d(RE, AA, CL, kind='cubic')
fCD = interp2d(RE, AA, CD, kind='cubic')

# vectorize lift and drag functions:
vector_fCL = np.vectorize(fCL)
vector_fCD = np.vectorize(fCD)

def getCl(re, aa):
    """ returns an array of lift coefficients """
    return vector_fCL(re, np.degrees(aa))

def getCd(re, aa):
    """ returns an array of drag coefficients """
    return vector_fCD(re, np.degrees(aa))


# -----------------------------------------------------------------------------
#
# previous settings to main computations
#
# -----------------------------------------------------------------------------
# angular positions in radians:
t = np.fromfunction(lambda j : (1.0 / 2.0 + j) * DT, (N, ))

# extended angular positions in radians:
p = np.linspace(t[0], t[-1], num=20 * N)

# coordinates of the ring:
x = -F * np.sin(t)
y = +F * np.cos(t)

# initialization of influence matrices:
cx = np.zeros((N, N), dtype=float)
cy = np.zeros((N, N), dtype=float)

# call the fill function:
fillMatrices(cx, cy, x, y, t, N, DT)

# wake matrices:
wn = np.zeros((N, N), dtype=float)
wt = np.zeros((N, N), dtype=float)

# fill the normal loads wake matrix (lower half):
right_index = (np.arange(N / 2, N)).astype(int)
left_index = (np.arange(N / 2 - 1, -1, -1)).astype(int)

wn[right_index, right_index] = 1.00
wn[right_index, left_index] = -1.00

# fill the tangential loads wake matrix (lower half):
Y = F * np.cos(t[N // 2 + 1: N - 1])

wt[right_index[1:-1], right_index[1:-1]] = -Y/np.sqrt(1.0 - Y ** 2)
wt[right_index[1:-1], left_index[1:-1]] = -Y/np.sqrt(1.0 - Y ** 2)

# the [1:-1] means that the head and tail are omitted due to the fact
# that Y would yield values greater than 1 in the poles, thereby
# leading to singularities (-Y/(1 - Y^2)^(1/2)).


# -----------------------------------------------------------------------------
#
# correction factor for velocities
#
# -----------------------------------------------------------------------------
def correction(qn, qt):
    """ computes the correction factors for the perturbation velocities """

    global t, p

    # empirical coefficients:
    k3, k2, k1, k0 = 0.0892, 0.0544, 0.2511, -0.0017

    # thrust as a function of the azimuth angle and the loads:
    thrust = qn * np.sin(t) + qt * np.cos(t)

    # interpolator function for the thrust:
    function = interp1d(t, thrust, kind='cubic')

    # vectorize the function so that it takes an array of angles:
    vector_function = np.vectorize(function)

    # thrust coefficient integrating according to phi:
    cth = simps(vector_function(p), p)

    # induction factor:
    a = k3 * (cth ** 3) + k2 * (cth ** 2) + (k1 * cth) + k0

    # correction factor:
    if a <= 0.15:
        ka = 1 / (1 - a)
    else:
        ka = (1 / (1 - a)) * (0.65 + 0.35 * exp(-4.5 * (a - 0.15)))

    return a, cth, ka

def correction2(q, qt):
    # thrust as a function of the azimuth angle and the loads:
    thrust = qn * np.sin(t) + qt * np.cos(t)

    # interpolator function for the thrust:
    function = interp1d(t, thrust, kind='cubic')

    # vectorize the function so that it takes an array of angles:
    vector_function = np.vectorize(function)

    # thrust coefficient integrating according to phi:
    cth = simps(vector_function(p), p)

    # induction factor
    if cth <= 0.96:
        a = (1 / 2) * (1 - sqrt(1 - cth))
        ka = 1 / (1 - a)
    else:
        a = (1 / 7) * (1 + 3 * sqrt(3.5 * cth - 3))
        ka = 18 * a / (7 * pow(a, 2) - 2 * a + 4)
        
    return a, cth, ka

# -----------------------------------------------------------------------------
#
# compute power coefficient
#
# -----------------------------------------------------------------------------
def power(qt, TSR):
    """ self explanatory """

    global t, p

    # make a function out of qt(t)
    function = interp1d(t, qt, kind='cubic')
    vector_function = np.vectorize(function)

    return -TSR * simps(vector_function(p), p)


# -----------------------------------------------------------------------------
#
# grid computation
#
# -----------------------------------------------------------------------------
X = np.arange(-2.0, 4.0, 0.05)
Y = np.linspace(2, -2, 100)

# mesh grid
gridX, gridY = np.meshgrid(X, Y)

# flattened grids
flatX = gridX.ravel()
flatY = gridY.ravel()

# perturbation velocities in the grid:
WX = np.zeros((flatX.size, ), dtype=float)
WY = np.zeros((flatX.size, ), dtype=float)
WW = np.zeros((flatX.size, ), dtype=float)


# -----------------------------------------------------------------------------
#
# main computations
#
# -----------------------------------------------------------------------------
# the operations are done element-wise.

# vx: dimensionless streamwise velocity
# vy: dimensionless cross-stream velocity
# vn: dimensionless normal velocity (normal to the chord)
# vt: dimensionless tangential velocity (tangent to the chord)
# vr: relative velocity
# aa: angle of attack
# re: local Reynolds number
# cl: lift coefficient
# cd: drag coefficient
# cn: normal force coefficient
# ct: tangent force coefficient
# qn: blades' normal force coefficient (one-revolution average)
# qt: blades' tangential force coefficient (one-revolution average)


# perturbation velocities:
wx = np.zeros((N, ), dtype=float)
wy = np.zeros((N, ), dtype=float)

its = 0

while True:

    vx = 1.0 + wx                                                       # vx
    vy = wy                                                             # vy
    vn = vx * np.sin(t) - vy * np.cos(t)                                # vn
    vt = vx * np.cos(t) + vy * np.sin(t) + TSR                          # vt
    vr = np.sqrt(vn**2 + vt**2)                                         # vr
    aa = np.arctan(vn / vt) - BPA                                       # aa
    re = REG * vr / (TSR * 1e6)                                         # re
    cl = getCl(re, aa)                                                  # cl
    cd = getCd(re, aa)                                                  # cd
    cn = cl * np.cos(aa) + cd * np.sin(aa)                              # cn
    ct = cl * np.sin(aa) - cd * np.cos(aa)                              # ct
    qn = (SOL / (2 * PI)) * (vr ** 2) * (cn * cos(BPA) - ct * sin(BPA)) # qt                          
    qt = -(SOL / (2 * PI)) * (vr ** 2) * (cn * sin(BPA) + ct * cos(BPA))# qn                             

    a, cth, ka = correction2(qn, qt)

    wxNew = (cx + wn).dot(qn) + (cy + wt).dot(qt)
    wyNew = cy.dot(qn) - cx.dot(qt)
    wxNew = wxNew * ka
    wyNew = wyNew * ka

    if wxNew[N // 4] > 0:
        wxNew = -1 * wxNew
        wyNew = -1 * wyNew

    if (np.allclose(wx, wxNew, rtol=E) and np.allclose(wy, wyNew, rtol=E)) or its > 100:
        break
    else:
        wx = K * wxNew + (1 - K) * wx
        wy = K * wyNew + (1 - K) * wy

    its += 1

# out of convergence loop:
print('{0:.<80}\n'.format('Convergence achieved '))


# -----------------------------------------------------------------------------
#
# print to console
#
# -----------------------------------------------------------------------------

# power coefficient:
cp = power(qt, TSR)

# freestream velocity:
u = (RAD * RPM * PI / 30) / TSR

# console report:
print('a : {0:.>8.2f}\n'.format(a))
print('ct: {0:.>8.2f}\n'.format(cth))
print('cp: {0:.>8.2f}\n'.format(cp))
print('ui: {0:.>8.2f}\n'.format(u))


# -----------------------------------------------------------------------------
#
# color maps
#
# -----------------------------------------------------------------------------
# make function out of qn:
qnFun = interp1d(t, qn, kind='quadratic')

# finer qn:
vector_qnFun = np.vectorize(qnFun)
qnFine = vector_qnFun(p)

# fill WX & WY:
WX = getWx(flatX, flatY, p, qnFine, WX)
WY = getWy(flatX, flatY, p, qnFine, WY)
WW = getWw(flatX, flatY, qnFun, WW)

# VX = WX + WW + 1
# VY = WY
WX += WW + 1

# velocity modulus:
V = np.sqrt(WX ** 2 + WY ** 2)

# resize flattened arrays:
V.resize(Y.size, X.size)
WX.resize(Y.size, X.size)
WY.resize(Y.size, X.size)
WW.resize(Y.size, X.size)

circle = Circle((0,0), 1.0, color='black', fill=False)
fig = plt.figure()

ax = fig.add_subplot(1,1,1)
imx = ax.imshow(V, interpolation='bicubic', extent=[-2, 4, -2, 2], cmap='jet')
ax.set_xlabel('$x/R$')
ax.set_ylabel('$y/R$')
ax.set_title('Velocity Magnitude [-]')
cbar = fig.colorbar(ax=ax, mappable=imx, orientation='vertical')
ax.add_artist(circle)

plt.show()




