import numpy as np
from copy import copy
import rbdl

pi = np.pi
cos = np.cos
sin = np.sin

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/majager.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.q[5] = min(max(self.q[5], 0.0), 0.3)
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq
def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine_majager(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]

    """
    # Longitudes (en metros)

    # Matrices DH (completar)
    T1 = dh(0.580,          q[0]+np.pi/2,   0.200,   np.pi/2)
    T2 = dh(0,              q[1]+np.pi*2/3,   0.800,   0)
    T3 = dh(0,              q[2]-np.pi*(1/6),     0.130,   np.pi/2)
    T4 = dh(0.765,          q[3]+np.pi,           0,       np.pi/2)
    T5 = dh(0,              q[4]+np.pi/2,   0,       np.pi/2)
    T6 = dh(0.280+q[5],   0,              0,       0)
    T7 = dh(0.0275+0.04,        q[6],           0,       0)
    # Efector final con respecto a la base
    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7
    return T


def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Alocacion de memoria
    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    T = fkine_majager(q)
    # Iteracion para la derivada de cada columna
    for i in range(7):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] += delta
        # Transformacion homogenea luego del incremento (q+dq)
        T_inc = fkine_majager(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[:,i] = (T_inc[:3,3]-T[:3,3])/delta

    return J


def ikine_majager(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0.
    Emplear el metodo de newton
    """
    epsilon = 0.001
    max_iter = 10000
    delta = 0.00001

    q = copy(q0)
    for i in range(max_iter):
        # Main loop
        # Calculate the error in position
        T = fkine_majager(q)
        x = T[0:3, 3]
        error = np.array(xdes - x)

        # Check if error is below threshold
        if np.linalg.norm(error) < epsilon:
            break

        # Calculate Jacobian matrix
        J = jacobian_position(q)

        # Use pseudo-inverse of Jacobian for iterative update
        delta_q = np.linalg.pinv(J) @ error

        # Update joint angles
        q += delta_q
        """
        q[1] = min(max(q[1], -3.3161255), 1.3963)
        q[2] = min(max(q[2], -0.7), 4.2)
        q[4] = min(max(q[3], -0.87266), 4.01426)
        """
        q[5] = min(max(q[5], 0.0), 0.3)

        # Check for convergence
        if np.linalg.norm(delta_q) < delta:
            break

    return q


def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    J = np.zeros((7, 7))
    # Implementar este Jacobiano aqui
    T = fkine_majager(q)
    R = TF2xyzquat(T)
    # Iteracion para la derivada de cada articulacion (columna)
    for i in range(7):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Calcular nuevamenta la transformacion homogenea e
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+delta)
        T_inc = fkine_majager(dq)
        R_inc = TF2xyzquat(T_inc)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        # J[0:a,i]=(T_inc[]-T[])/delta
        J[:, i] = (R_inc - R) / delta

    return J



def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        quat[0] = 0.5 * np.sqrt(trace + 1.0)
        s = 1.0 / (4.0 * quat[0])
        quat[1] = (R[2,1] - R[1,2]) * s
        quat[2] = (R[0,2] - R[2,0]) * s
        quat[3] = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            quat[1] = 0.5 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            s = 1.0 / (4.0 * quat[1])
            quat[0] = (R[2,1] - R[1,2]) * s
            quat[2] = (R[1,0] + R[0,1]) * s
            quat[3] = (R[0,2] + R[2,0]) * s
        elif R[1,1] > R[2,2]:
            quat[2] = 0.5 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            s = 1.0 / (4.0 * quat[2])
            quat[0] = (R[0,2] - R[2,0]) * s
            quat[1] = (R[1,0] + R[0,1]) * s
            quat[3] = (R[2,1] + R[1,2]) * s
        else:
            quat[3] = 0.5 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            s = 1.0 / (4.0 * quat[3])
            quat[0] = (R[1,0] - R[0,1]) * s
            quat[1] = (R[0,2] + R[2,0]) * s
            quat[2] = (R[2,1] + R[1,2]) * s

    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R

def rotx(ang): #Rotacion en x
    Rx = np.array([[1, 0, 0],
                   [0, cos(ang), -sin(ang)],
                   [0, sin(ang), cos(ang)]])
    return Rx

def roty(ang): #Rotacion en y
    Ry = np.array([[cos(ang), 0, sin(ang)],
                   [0, 1, 0],
                   [-sin(ang), 0, cos(ang)]])
    return Ry

def rotz(ang): #Rotacion en z
    Rz = np.array([[cos(ang), -sin(ang), 0],
                   [sin(ang), cos(ang), 0],
                   [0,0,1]])
    return Rz
