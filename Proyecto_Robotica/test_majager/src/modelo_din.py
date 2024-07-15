import rbdl
import numpy as np

if __name__ == '__main__':

  # Lectura del modelo del robot a partir de URDF (parsing)
  modelo = rbdl.loadModel('../urdf/majager.urdf')
  # Grados de libertad
  ndof = modelo.q_size
  print("ndof: \n", ndof)

  # Configuracion articular
  q = np.array([0.5, 0.2, 0.3, 0.8, 0.5, 0.6, 0])
  # Velocidad articular
  dq = np.array([0.8, 0.7, 0.8, 0.6, 0.9, 1.0, 0])
  # Aceleracion articular
  ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5, 0])
  
  # Arrays numpy
  zeros = np.zeros(ndof)          # Vector de ceros
  tau   = np.zeros(ndof)          # Para torque
  g     = np.zeros(ndof)          # Para la gravedad
  c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
  M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
  e     = np.eye(ndof)               # Vector identidad
  
  # Torque dada la configuracion del robot
  rbdl.InverseDynamics(modelo, q, dq, ddq, tau)

  rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
  rbdl.InverseDynamics(modelo, q, dq, zeros, c)
  c = c-g
  m = np.zeros(ndof)
  for i in range(ndof):
    rbdl.InverseDynamics(modelo, q, zeros, e[i, :], m)
    M[i, :] = m - g

  np.set_printoptions(suppress=True)
  print("M: \n", np.round(M,3))
  print("c: \n", np.round(c,3))
  print("g: \n", np.round(g,3))

  b2 = np.zeros(ndof)          # Para efectos no lineales
  M2 = np.zeros([ndof, ndof])  # Para matriz de inercia

  rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)
  print("M2: \n", np.round(M2,3))
  rbdl.NonlinearEffects(modelo, q, dq, b2)
  print("b2: \n", np.round(b2, 3))
  print("c+g: \n", np.round(c+g, 3))

  t=M@ddq+b2
  print("tau: \n", np.round(tau, 3))
  print("tau2: \n", np.round(t, 3))

