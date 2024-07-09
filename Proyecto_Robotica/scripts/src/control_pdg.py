#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from lab5functions import *
from roslib import packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rbdl

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
  # Velocidad inicial
  dq = np.array([0., 0., 0., 0., 0., 0., 0])
  # Configuracion articular deseada
  qdes = np.array([0, -1.2, 1.2, 0, 0, 0, 0])
  # =============================================================
  
  # Posicion resultante de la configuracion articular deseada
  xdes = fkine_kr20(qdes)[0:3,3]
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
  Kp = 10*np.eye(ndof)
  Kd = 20*np.eye(ndof)
  
  # Bucle de ejecucion continua
  t = 0.0

  zeros = np.zeros(ndof)
  g = np.zeros(ndof)  # Para la gravedad

  while not rospy.is_shutdown():
  
    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    x = fkine_kr20(q)[0:3,3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # Almacenamiento de datos
    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+'\n ')
    fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+'\n ')

    # ----------------------------
    # Control dinamico (COMPLETAR)
    # ----------------------------
    u = np.zeros(ndof)   # Reemplazar por la ley de control
    rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
    u = g + Kp@(qdes-q)-Kd@dq
    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    q_2 = np.append(q, [0.03, 0.03])
    jstate.position = q_2
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

  fqact.close()
  fqdes.close()
  fxact.close()
  fxdes.close()



