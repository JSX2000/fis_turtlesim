import numpy as np
import skfuzzy as sk
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

def superficie(index_ang, index_dis, option):
    output = np.zeros_like(out_ang)
    suma, area  = 0, 0
    R = [[min(ang[index_ang], dis[index_dis]) for ang in angulo] for dis in distancia]

    if option == 1:
        reglas = [max(R[0][0], R[1][0], R[0][1], R[0][2], R[0][3], R[0][4], R[1][4]),
                  max(R[2][0], R[1][1], R[1][2], R[1][3], R[2][4]),
                  max(R[3][0], R[4][0], R[2][1], R[2][2], R[2][3], R[3][4], R[4][4]),
                  max(R[3][1], R[4][1], R[3][2], R[3][3], R[4][3]),
                  R[4][2]]
        
    else:
        reglas = [max(R[0][0], R[1][0], R[2][0]),
                  max(R[3][0], R[4][0], R[0][1], R[1][1], R[2][1], R[3][1], R[4][1]),
                  max(R[0][2], R[1][2], R[2][2], R[3][2], R[4][2]),
                  max(R[0][3], R[1][3], R[2][3], R[3][3], R[4][3], R[3][4], R[4][4]),
                  max(R[0][4], R[1][4], R[2][4])]

    for num in range(len(out_vel)): #out_vel grafica
        output[num] = max([min(reglas[mf_index], lin_final[mf_index][num]) for mf_index in range(len(reglas))])
        suma += out_vel[num] * output[num]
        area += output[num]
    return suma/area
    

#Defining the Numpy array for Tip Quality
in_ang = np.arange(-180, 180, 1)
in_dis = np.arange(0, 16.15, 0.1)
out_vel = np.arange(0, 4.1, 0.5) #lineal
out_ang = np.arange(-2.0, 2.1, 0.5) #angular

x = int(input('Variable x (-180, 180):\n'))
y = int(input('Variable y (0, 16):\n'))

x = np.where(in_ang == x)
y = np.where(in_dis == y)

#Defining the Numpy array for curve membership functions
ext_left = sk.sigmf(in_ang, -140, -0.1)
med_left = sk.gbellmf(in_ang, 25,1.5,-90)
cent = sk.gbellmf(in_ang, 25,1.5,0)
med_rig = sk.gbellmf(in_ang, 25,1.5,90)
ext_rig = sk.sigmf(in_ang, 140, 0.1)
angulo = [ext_left, med_left, cent, med_rig, ext_rig]

near = sk.gbellmf(in_dis, 1,1.5,0)
short = sk.gbellmf(in_dis, 1,1.5,4)
moderate = sk.gbellmf(in_dis, 1,1.5,8)
long = sk.gbellmf(in_dis, 1,1.5,12)
far = sk.gbellmf(in_dis, 1,1.5,16)
distancia = [near, short, moderate, long, far]

l_low = sk.trimf(out_vel, [0, 0, 1])
m_low = sk.trimf(out_vel, [0, 1, 2])
med = sk.trimf(out_vel, [1, 2, 3])
m_high = sk.trimf(out_vel, [2, 3, 4])
high = sk.trimf(out_vel, [3, 4, 4])
lin_final = [l_low, m_low, med, m_high, high]

h_cw = sk.trimf(out_ang, [-2, -2, -1])
m_cw = sk.trimf(out_ang, [-2, -1, 0])
low = sk.trimf(out_ang, [-1, 0, 1])
m_ccw = sk.trimf(out_ang, [0, 1, 2])
h_ccw = sk.trimf(out_ang, [1, 2, 2])
ang_final = [h_cw, m_cw, low, m_ccw, h_ccw]

R1 = min(ext_left[x], near[y])
R2 = min(ext_left[x], short[y])
R3 = min(ext_left[x], moderate[y])
R4 = min(ext_left[x], long[y])
R5 = min(ext_left[x], far[y])

R6 = min(med_left[x], near[y])
R7 = min(med_left[x], short[y])
R8 = min(med_left[x], moderate[y])
R9 = min(med_left[x], long[y])
R10 = min(med_left[x], far[y])

R11 = min(cent[x], near[y])
R12 = min(cent[x], short[y])
R13 = min(cent[x], moderate[y])
R14 = min(cent[x], long[y])
R15 = min(cent[x], far[y])

R16 = min(med_rig[x], near[y])
R17 = min(med_rig[x], short[y])
R18 = min(med_rig[x], moderate[y])
R19 = min(med_rig[x], long[y])
R20 = min(med_rig[x], far[y])

R21 = min(ext_rig[x], near[y])
R22 = min(ext_rig[x], short[y])
R23 = min(ext_rig[x], moderate[y])
R24 = min(ext_rig[x], long[y])
R25 = min(ext_rig[x], far[y])

l = max(R1, R2, R6, R11, R16, R21, R22)
m_l = max(R3, R7, R12, R17, R23)
m = max(R4, R5, R8, R13, R18, R24, R25)
m_h = max(R9, R10, R14, R19, R20)
h = R15

a_i = max(R1, R2, R3)
m_i = max(R4, R5, R6, R7, R8, R9, R10)
cero = max(R11, R12, R13, R14, R15)
m_d = max(R16, R17, R18, R19, R20, R24, R25)
a_d = max(R21, R22, R23)

v_z1 = np.fmin(l, l_low)
v_z2 = np.fmin(m_l, m_low)
v_z3 = np.fmin(m, med)
v_z4 = np.fmin(m_h, m_high)
v_z5 = np.fmin(h, high)

w_z1 = np.fmin(a_i, h_cw)
w_z2 = np.fmin(m_i, m_cw)
w_z3 = np.fmin(cero, low)
w_z4 = np.fmin(m_d, m_ccw)
w_z5 = np.fmin(a_d, h_ccw)

aggregated_v = np.fmax(v_z1, np.fmax(v_z2, np.fmax(v_z3, np.fmax(v_z4, v_z5))))
aggregated_w = np.fmax(w_z1, np.fmax(w_z2, np.fmax(w_z3, np.fmax(w_z4, w_z5))))

plt.subplot(2, 3, 1)
plt.plot(in_ang,ext_left)
plt.plot(in_ang,med_left)
plt.plot(in_ang,cent)
plt.plot(in_ang,med_rig)
plt.plot(in_ang,ext_rig)
plt.ylabel('\u03BC')
plt.xlabel('\u03B8 [rad]')
plt.legend(['ext left', 'med left', 'center', 'med right', 'ext right'])
plt.title("Posición angular")

plt.subplot(2, 3, 4)
plt.plot(in_dis,near)
plt.plot(in_dis,short)
plt.plot(in_dis,moderate)
plt.plot(in_dis,long)
plt.plot(in_dis,far)
plt.ylabel('\u03BC')
plt.xlabel('m')
plt.legend(['near', 'short', 'moderate', 'long', 'far'])
plt.title("Distancia")

plt.subplot(2, 3, 2)
plt.plot(out_vel,l_low)
plt.plot(out_vel,m_low)
plt.plot(out_vel,med)
plt.plot(out_vel,m_high)
plt.plot(out_vel,high)
plt.ylabel('\u03BC')
plt.xlabel('m')
plt.legend(['low', 'm_low', 'med', 'm_high', 'high'])
plt.title("Velocidad Lineal")

plt.subplot(2, 3, 5)
plt.plot(out_ang,h_cw)
plt.plot(out_ang,m_cw)
plt.plot(out_ang,low)
plt.plot(out_ang,m_ccw)
plt.plot(out_ang,h_ccw)
plt.ylabel('\u03BC')
plt.xlabel('km/h')
plt.legend(['h cw', 'm cw', 'low', 'm ccw', 'h ccw'])
plt.title("Velocidad angular")

plt.subplot(2, 3, 3)
plt.plot(out_vel, aggregated_v)
plt.ylim([0.0, 1.0])
plt.ylabel('\u03BC')
plt.xlabel('m/s')
plt.legend(['resultado'])
plt.title("Velocidad lineal final")

plt.subplot(2, 3, 6)
plt.plot(out_ang, aggregated_w)
plt.ylim([0.0, 1.0])
plt.ylabel('\u03BC')
plt.xlabel('m/s')
plt.legend(['resultado'])
plt.title("Velocidad angular final")

plt.show()

plt.figure(1)
ax = plt.axes(projection = '3d')
x_3d_lin, y_3d_lin = np.meshgrid(in_dis, in_ang)
z_3d_lin = np.zeros((len(in_ang), len(in_dis)))
for i in range(len(in_dis)):
    for j in range(len(in_ang)):
        z_3d_lin[j, i] = superficie(np.round((in_ang[j] + 180)), int(in_dis[i] * 10), 1)
        
ax.plot_surface(x_3d_lin, y_3d_lin, z_3d_lin, rstride = 1, cstride = 1, cmap = 'inferno', edgecolor = 'none')
ax.set_title("Superficie de Control velocidad lineal")
ax.set_ylabel("Diferencia angular")
ax.set_xlabel("Diferencia distancia")
plt.show()

plt.figure(2)
ax = plt.axes(projection = '3d')
x_3d_ang, y_3d_ang = np.meshgrid(in_dis, in_ang)
z_3d_ang = np.zeros((len(in_ang), len(in_dis)))
for i in range(len(in_dis)):
    for j in range(len(in_ang)):
        z_3d_ang[j, i] = superficie(np.round((in_ang[j] + 180)), int(in_dis[i] * 10), 2)
        
ax.plot_surface(x_3d_ang, y_3d_ang, z_3d_ang, rstride = 1, cstride = 1, cmap = 'inferno', edgecolor = 'none')
ax.set_title("Superficie de Control velocidad angular")
ax.set_ylabel("Diferencia angular")
ax.set_xlabel("Diferencia distancia")
ax.invert_xaxis()
plt.show()

with open("surfaceZ_lin.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(z_3d_lin)
    
with open("surfaceZ_ang.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(z_3d_ang)
    
with open("surfaceX.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(x_3d_lin)
    
with open("surfaceY.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(y_3d_lin)
