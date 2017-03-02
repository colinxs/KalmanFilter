import numpy as np
import scipy as sp
from numpy.linalg.linalg import transpose as T
from numpy.linalg.linalg import inv as I
from matplotlib import pyplot as plt


class LinearKalmanFilter:

    def __init__ (self, A, B, P0, Q, R, H, x0):
        # all matricies and arrays are numpy ndarrays

        # convention: current state is k-1 i.e. last full prediction

        self._A = A # state transition matrix, relates previous state to current state (nxn)
        self._B = B # control matrix, relates control vector to state vector (if u is px1 then B is nxp)
        self._P_k_1 = P0 # initial a posteriori estimate error covariance matrix (nxn)
        self._Q = Q # process error covariance (nxn)
        self._R = R # measurement error covariance (if z is mx1, then R is mxm)
        self._H = H # observation matrix relates state vector, x, to measurement vector z (if z mx1, then H is mxn)
        self._x_k_1 = x0 # initial state estimation (nx1)

    def get_current_state(self):
        return self._x_k_1

    # u_k_1 = control vector (
    def predict(self, u_k_1):
        # project state ahead
        a_priori_x_k = self._A @ self._x_k_1 + self._B @ u_k_1

        # project error covariance ahead
        a_priori_P_k = self._A @ self._P_k_1 @ T(self._A) + self._Q

        return (a_priori_x_k, a_priori_P_k)

    def correct(self, a_priori_x_k, a_priori_P_k, z_k):
        # compute kalman gain (nxm)
        gain = a_priori_P_k @ T(self._H) @ I((self._H @ a_priori_P_k @ T(self.H) + self._R))

        # update estimate with measurement z_k
        x_k = a_priori_x_k + gain @ (z_k - self._H @ a_priori_x_k)

        # update the error covariance
        P_k = (np.identity(self._P_k_1.shape) - gain @ self._H) @ a_priori_P_k

        return (x_k, P_k)

    def step(self, u_k_1, z_k):
        a_priori_x_k, a_priori_P_k = self.predict(u_k_1)
        self._x_k_1, self._P_k_1 = self.correct(a_priori_x_k, a_priori_P_k, z_k)


def main():


    theta = 60.0 # firing angle(degrees)
    v0 = 10.0 # initial speed (m/s)
    
    # x constraints
    x0 = 0
    v0x = v0 * np.cos(np.deg2rad(theta))
    a0x = 0

    # y constrains
    y0 = 0.0
    v0y = v0 * np.sin(np.deg2rad(theta))
    a0y = -9.81


    n = 100 # number of data samples
    t0 = 0 # start time
    tf = (-v0y - np.sqrt(v0y ** 2 - 4 * a0y/2 * y0)) / a0y # end time
    t = np.linspace(t0, tf, n+1) # time  vector

    x = x0 + t * v0x + 1 / 2 * a0x * t ** 2
    y = y0 + t * v0y + 1 / 2 * a0y * t ** 2

    # add noise
    sigma = 0.2
    y_noisy = y + np.random.normal(0, sigma, n + 1)
    x_noisy = x + np.random.normal(0, sigma, n + 1)


    # attempt Kalman Filter
    A = 
    kf = LinearKalmanFilter(A, B, P0, Q, R, H, x0)

    plt.figure()
    plt.scatter(x_noisy, y_noisy, label="Measured")
    plt.plot(x, y, '--g', label="True")
    #print([min(min(x_noisy), min(x)) - 0.5, max(max(x_noisy), max(x)) + 0.5, min(min(y_noisy), min(y)) - 0.5, max(max(y_noisy), max(y)) + 0.5])
    plt.axis([min(min(x_noisy), min(x)) - 0.5, max(max(x_noisy), max(x)) + 0.5, min(min(y_noisy), min(y)) - 0.5, max(max(y_noisy), max(y)) + 0.5])
    plt.title('True vs Measured Position of a Projectile with Gaussian noise (\sigma=0.5)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
