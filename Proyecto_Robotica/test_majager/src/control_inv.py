#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from functions import *
from roslib import packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rbdl
global x0, xd
x0=np.array([0, 0, 0])
xd=np.array([0, 0, 0])
if __name__ == '__main__':

  rospy.init_node("control_pdg")
  pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
  bmarker_actual  = BallMarker(color['RED'])
  bmarker_deseado = BallMarker(color['GREEN'])
  # Archivos donde se almacenara los datos
  fqact = open("/tmp/qactual.txt", "w")
  fqdes = open("/tmp/qdeseado.txt", "w")
  fxact = open("/tmp/xactual.txt", "w")
  fxdes = open("/tmp/xdeseado.txt", "w")
  
  # Nombres de las articulaciones
  jnames = ['Base_to_Link1', 'Link1_to_Link2', 'Link2_to_Link3', 'Link3_to_Link4', 'Link4_to_Link5', 'Link5_to_Link6',
            'Link6_to_Base_Gripper', 'Base_Gripper_to_Gripper_Right', 'Base_Gripper_to_Gripper_Left']
  # Objeto (mensaje) de tipo JointState
  jstate = JointState()
  # Valores del mensaje
  jstate.header.stamp = rospy.Time.now()
  jstate.name = jnames
  
  # =============================================================
  # Configuracion articular inicial (en radianes)
  q = np.array([0.0, 0, 0, 0, 0, 0, 0])
  x0 = fkine_majager(q)[0:3, 3]
  # Velocidad inicial
  dq = np.array([0., 0., 0., 0., 0., 0., 0])
  # Configuracion articular deseada
  qd = np.array([5, 1.3963, 0.2, 3, 3, 0.1, 3])
  "qd = np.array([0, 0, 1, 1, -0.9, 0.05, 0])"
  "qd = np.array([1.2, -1.1, 0.85, 1.14, -0.5, 0.1, 1])"
  # =============================================================
  dqd = np.array([0., 0., 0., 0., 0., 0., 0])
  ddqd = np.array([0., 0., 0., 0., 0., 0., 0])
  ddq = np.array([0., 0., 0., 0., 0., 0., 0])
  # Posicion resultante de la configuracion articular deseada
  xd = fkine_majager(qd)[0:3, 3]
  # Copiar la configuracion articular en el mensaje a ser publicado
  q_2 = np.append(q, [0.03, 0.03])
  jstate.position = q_2
  pub.publish(jstate)

  # Modelo RBDL
  modelo = rbdl.loadModel('../urdf/majager.urdf')
  ndof   = modelo.q_size     # Grados de libertad
  print("ndof: \n", ndof)
  # Frecuencia del envio (en Hz)
  freq = 20
  dt = 1.0/freq
  rate = rospy.Rate(freq)

  # Simulador dinamico del robot
  robot = Robot(q, dq, ndof, dt)

  # Se definen las ganancias del controlador
  """
  Kp = 4*np.eye(ndof)
  Kd = 11*np.eye(ndof)
  """
  Kp = 0.6*np.diag(np.array([6,6,11,2,2,5,1]))
  Kd = 2*np.sqrt(Kp)

  cg = np.zeros(ndof)  # Para efectos no lineales
  Mq = np.zeros([ndof, ndof])  # Para matriz de inercia



  # Bucle de ejecucion continua
  t = 0.0

  zeros = np.zeros(ndof)
  g = np.zeros(ndof)  # Para la gravedad
  dq0=np.array([0., 0., 0., 0., 0., 0., 0])

  while not rospy.is_shutdown():
    dq0 = dq
    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = fkine_majager(q)[0:3, 3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()
    if(t==0):
      ddq=0
    else:
      ddq=(dq-dq0)/t

    # Almacenamiento de datos
    fxact.write(str(t) + ' ' + str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + '\n')
    fxdes.write(str(t) + ' ' + str(xd[0]) + ' ' + str(xd[1]) + ' ' + str(xd[2]) + '\n')
    fqact.write(str(t) + ' ' + str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + str(q[3]) + ' ' + str(
      q[4]) + ' ' + str(q[5]) + ' ' + str(q[6]) + '\n ')
    fqdes.write(
      str(t) + ' ' + str(qd[0]) + ' ' + str(qd[1]) + ' ' + str(qd[2]) + ' ' + str(qd[3]) + ' ' + str(
        qd[4]) + ' ' + str(qd[5]) + ' ' + str(qd[6]) + '\n ')
    # ----------------------------
    # Control dinamico (COMPLETAR)

    rbdl.CompositeRigidBodyAlgorithm(modelo, q, Mq)
    rbdl.NonlinearEffects(modelo, q, dq, cg)

    # ----------------------------
    u = np.zeros(ndof)   # Reemplazar por la ley de control
    rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
    u = Mq@( ddqd-ddq + Kd@(dqd-dq) + Kp@(qd-q) ) + cg
    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    q_2 = np.append(q, [0.03, 0.03])
    jstate.position = q_2
    pub.publish(jstate)
    bmarker_deseado.xyz(xd)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

  print('ending motion ...')
  print("\n")
  print("Articulación: " + "q1: " + str(round(q[0],4)) + " rad, q2: " + str(round(q[1],4)) + " rad, q3: " + str(round(q[2],4)) + " rad")
  print("              q4: " + str(round(q[3],4)) + " rad, q5: " + str(round(q[4],4)) + " rad, q6: " + str(round(q[5],4)) + " m, q7: " + str(
    round(q[6],4)) + " rad")
  print("Articulación deseada: " + "q1: " + str(qd[0]) + " rad, q2: " + str(qd[1]) + " rad, q3: " + str(qd[2]) + " rad")
  print("              q4: " + str(qd[3]) + " rad, q5: " + str(qd[4]) + " rad, q6: " + str(qd[5]) + " m, q7: " + str(
    qd[6]) + " rad")
  print("Posición deseada: " + "X: " + str(round(xd[0],4)) + ", Y: " + str(round(xd[1],4)) + ", Z: " + str(round(xd[2],4)))
  print(
    "Posición final: " + "X: " + str(round(x[0],4)) + ", Y: " + str(round(x[1],4)) + ", Z: " + str(round(x[2],4)))
  print("\n")
  fqact.close()
  fqdes.close()
  fxact.close()
  fxdes.close()

# Leer datos desde el archivo
with open("/tmp/xactual.txt", 'r') as file:
  lines = file.readlines()
with open("/tmp/xdeseado.txt", 'r') as file:
  lines2 = file.readlines()
# Inicializar listas para almacenar las coordenadas
x_data = []
y_data = []
z_data = []
t_data = []

xd_data = []
yd_data = []
zd_data = []
td_data = []
# Procesar cada línea del archivo
for line in lines:
  # Dividir la línea en valores de x, y, y z
  data = line.split()
  # Convertir los valores a números flotantes y agregarlos a las listas correspondientes
  t_data.append(float(data[0]))
  x_data.append(float(data[1]))
  y_data.append(float(data[2]))
  z_data.append(float(data[3]))

for line2 in lines2:
  # Dividir la línea en valores de x, y, y z
  data2 = line2.split()
  # Convertir los valores a números flotantes y agregarlos a las listas correspondientes
  td_data.append(float(data2[0]))
  xd_data.append(float(data2[1]))
  yd_data.append(float(data2[2]))
  zd_data.append(float(data2[3]))
# Convertir las listas de coordenadas a arrays de NumPy

x_data = np.array(x_data)
y_data = np.array(y_data)
z_data = np.array(z_data)
t_data = np.array(t_data)
xd_data = np.array(xd_data)
yd_data = np.array(yd_data)
zd_data = np.array(zd_data)
td_data = np.array(td_data)

# Crear la gráfica
fig, axs = plt.subplots(3, 1, figsize=(10, 10), edgecolor='black')
formatter1 = ticker.ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((-1, 1))

axs[0].plot(t_data, x_data, label='x')
axs[0].plot(td_data, xd_data, linestyle='--', label='x_ref')
axs[0].set_xlabel('t (s)')
axs[0].set_ylabel('posicion (m)')
axs[0].set_title('Gráfica de Posicion x')
axs[0].grid(True)
axs[0].legend(loc="best")
axs[0].xaxis.set_major_formatter(formatter1)
axs[0].yaxis.set_major_formatter(formatter1)

axs[1].plot(t_data, y_data, label='y')
axs[1].plot(td_data, yd_data, linestyle='--', label='y_ref')
axs[1].set_xlabel('t (s)')
axs[1].set_ylabel('posicion (m)')
axs[1].set_title('Gráfica de Posicion y')
axs[1].grid(True)
axs[1].legend(loc="best")
"axs[1].xaxis.set_major_formatter(formatter1)"
"axs[1].yaxis.set_major_formatter(formatter1)"

axs[2].plot(t_data, z_data, label='z')
axs[2].plot(td_data, zd_data, linestyle='--', label='z_ref')
axs[2].set_xlabel('t (s)')
axs[2].set_ylabel('posicion (m)')
axs[2].set_title('Gráfica de Posicion z')
axs[2].grid(True)
axs[2].legend(loc="best")
"axs[2].xaxis.set_major_formatter(formatter1)"
"axs[2].yaxis.set_major_formatter(formatter1)"
plt.tight_layout()
# Etiquetas y título

with open("/tmp/qactual.txt", 'r') as file:
  lines3 = file.readlines()
lines3 = lines3[:-1]

with open("/tmp/qdeseado.txt", 'r') as file:
  lines4 = file.readlines()
lines4 = lines4[:-1]

# Inicializar listas para almacenar las coordenadas
q1_data = []
q2_data = []
q3_data = []
q4_data = []
q5_data = []
q6_data = []
q7_data = []

t_data = []

# Inicializar listas para almacenar las coordenadas
q1d_data = []
q2d_data = []
q3d_data = []
q4d_data = []
q5d_data = []
q6d_data = []
q7d_data = []

td_data = []
# Procesar cada línea del archivo
for line in lines3:
  # Dividir la línea en valores de x, y, y z
  data = line.split()
  # Convertir los valores a números flotantes y agregarlos a las listas correspondientes
  t_data.append(float(data[0]))
  q1_data.append(float(data[1]))
  q2_data.append(float(data[2]))
  q3_data.append(float(data[3]))
  q4_data.append(float(data[4]))
  q5_data.append(float(data[5]))
  q6_data.append(float(data[6]))
  q7_data.append(float(data[7]))

  # Procesar cada línea del archivo
for line in lines4:
  # Dividir la línea en valores de x, y, y z
  data = line.split()
  # Convertir los valores a números flotantes y agregarlos a las listas correspondientes
  td_data.append(float(data[0]))
  q1d_data.append(float(data[1]))
  q2d_data.append(float(data[2]))
  q3d_data.append(float(data[3]))
  q4d_data.append(float(data[4]))
  q5d_data.append(float(data[5]))
  q6d_data.append(float(data[6]))
  q7d_data.append(float(data[7]))

# Convertir las listas de coordenadas a arrays de NumPy

q1_data = np.array(q1_data)
q2_data = np.array(q2_data)
q3_data = np.array(q3_data)
q4_data = np.array(q4_data)
q5_data = np.array(q5_data)
q6_data = np.array(q6_data)
q7_data = np.array(q7_data)
t_data = np.array(t_data)

q1d_data = np.array(q1d_data)
q2d_data = np.array(q2d_data)
q3d_data = np.array(q3d_data)
q4d_data = np.array(q4d_data)
q5d_data = np.array(q5d_data)
q6d_data = np.array(q6d_data)
q7d_data = np.array(q7d_data)
td_data = np.array(td_data)

# Crear la gráfica
fig2, axs2 = plt.subplots(3, 3, figsize=(10, 10), edgecolor='black')

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
for ax in axs2.flat:
  ax.xaxis.set_major_formatter(formatter)
  ax.yaxis.set_major_formatter(formatter)

axs2[0, 0].plot(t_data, q1_data, label='q1')
axs2[0, 0].plot(td_data, q1d_data, linestyle='--', label='q1_ref')
axs2[0, 0].set_xlabel('t (s)')
axs2[0, 0].set_ylabel('q (rad)')
axs2[0, 0].set_title('Gráfica de q1')
axs2[0, 0].grid(True)
axs2[0, 0].legend(loc="best")

axs2[0, 1].plot(t_data, q2_data, label='q2')
axs2[0, 1].plot(td_data, q2d_data, linestyle='--', label='q2_ref')
axs2[0, 1].set_xlabel('t (s)')
axs2[0, 1].set_ylabel('q (rad)')
axs2[0, 1].set_title('Gráfica de q2')
axs2[0, 1].grid(True)
axs2[0, 1].legend(loc="best")

axs2[0, 2].plot(t_data, q3_data, label='q3')
axs2[0, 2].plot(td_data, q3d_data, linestyle='--', label='q3_ref')
axs2[0, 2].set_xlabel('t (s)')
axs2[0, 2].set_ylabel('q (rad)')
axs2[0, 2].set_title('Gráfica de q3')
axs2[0, 2].grid(True)
axs2[0, 2].legend(loc="best")

axs2[1, 0].plot(t_data, q4_data, label='q4')
axs2[1, 0].plot(td_data, q4d_data, linestyle='--', label='q4_ref')
axs2[1, 0].set_xlabel('t (s)')
axs2[1, 0].set_ylabel('q (rad)')
axs2[1, 0].set_title('Gráfica de q4')
axs2[1, 0].grid(True)
axs2[1, 0].legend(loc="best")

axs2[1, 1].plot(t_data, q5_data, label='q5')
axs2[1, 1].plot(td_data, q5d_data, linestyle='--', label='q5_ref')
axs2[1, 1].set_xlabel('t (s)')
axs2[1, 1].set_ylabel('q (rad)')
axs2[1, 1].set_title('Gráfica de q5')
axs2[1, 1].grid(True)
axs2[1, 1].legend(loc="best")

axs2[1, 2].plot(t_data, q6_data, label='q6')
axs2[1, 2].plot(td_data, q6d_data, linestyle='--', label='q6_ref')
axs2[1, 2].set_xlabel('t (s)')
axs2[1, 2].set_ylabel('q (rad)')
axs2[1, 2].set_title('Gráfica de q6')
axs2[1, 2].grid(True)
axs2[1, 2].legend(loc="best")

axs2[2, 0].axis('off')

axs2[2, 1].plot(t_data, q7_data, label='q7')
axs2[2,1].plot(td_data, q7d_data, linestyle='--', label='q7_ref')
axs2[2, 1].set_xlabel('t (s)')
axs2[2, 1].set_ylabel('q (rad)')
axs2[2, 1].set_title('Gráfica de q7')
axs2[2, 1].grid(True)
axs2[2, 1].legend(loc="best")

axs2[2, 2].axis('off')

plt.tight_layout()

# Crear la figura y el espacio 3D
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos en 3D
ax.plot(x_data, y_data, z_data)
ax.scatter(x0[0], x0[1], x0[2], color='red', label='Initial Position')
ax.scatter(xd[0], xd[1], xd[2], color='green', label='Expected Position')

# Configurar etiquetas
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Posición del efector final 3D')

x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()

x_range = x_limits[1] - x_limits[0]
y_range = y_limits[1] - y_limits[0]
z_range = z_limits[1] - z_limits[0]

max_range = max(x_range, y_range, z_range)

mid_x = (x_limits[1] + x_limits[0]) * 0.5
mid_y = (y_limits[1] + y_limits[0]) * 0.5
mid_z = (z_limits[1] + z_limits[0]) * 0.5

ax.set_xlim3d([mid_x - max_range / 2, mid_x + max_range / 2])
ax.set_ylim3d([mid_y - max_range / 2, mid_y + max_range / 2])
ax.set_zlim3d([mid_z - max_range / 2, mid_z + max_range / 2])
# Mostrar la gráfica
plt.show()



