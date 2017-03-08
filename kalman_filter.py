import numpy as np
import scipy.constants
from numpy.linalg.linalg import transpose as T
from numpy.linalg.linalg import inv as I
from matplotlib import pyplot as plt

# http://www.cs.yale.edu/homes/hudak-paul/CS474S01/kalman.pdf

class LinearKalmanFilter:

    def __init__ (self, A, B, P0, Q, R, H, x0, z0):
        # all matricies and arrays are numpy ndarrays

        # convention: current state is k-1 i.e. last full prediction

        self._A = A # state transition matrix, relates previous state to current state (nxn)
        self._B = B # control matrix, relates control vector to state vector (if u is px1 then B is nxp)
        self._Q = Q # process error covariance (nxn)
        self._R = R # measurement error covariance (if z is mx1, then R is mxm)
        self._H = H # observation matrix relates state vector, x, to measurement vector z (if z mx1, then H is mxn)
        self._x_k, self._P_k = self._correct(x0, P0, z0)


    def get_current_state(self):
        return self._x_k

    # u_k_1 = control vector (
    def _predict(self, u_k):
        # project state ahead
        a_priori_x_k_1 = self._A @ self._x_k + self._B @ u_k

        # project error covariance ahead
        a_priori_P_k_1 = self._A @ self._P_k @ T(self._A) + self._Q

        return (a_priori_x_k_1, a_priori_P_k_1)

    def _correct(self, a_priori_x_k, a_priori_P_k, z_k):
        # compute kalman gain (nxm)
        gain = a_priori_P_k @ T(self._H) @ I((self._H @ a_priori_P_k @ T(self._H) + self._R))
        # print(gain)

        # update estimate with measurement z_k
        x_k = a_priori_x_k + gain @ (z_k - self._H @ a_priori_x_k)

        # update the error covariance
        P_k = (np.identity(a_priori_P_k.shape[0]) - gain @ self._H) @ a_priori_P_k

        return (x_k, P_k)

    def step(self, u_k_1, z_k):
        a_priori_x_k_1, a_priori_P_k_1 = self._predict(u_k_1)
        self._x_k, self._P_k = self._correct(a_priori_x_k_1, a_priori_P_k_1, z_k) # k++
        return (self._x_k, a_priori_x_k_1, self._P_k, a_priori_P_k_1)

    def _filter_offline(self, u, z):
        x_k_vals = np.zeros((z.shape[0] + 1, self._x_k.shape[0]))
        x_k_1_vals = np.zeros((z.shape[0], self._x_k.shape[0]))
        P_k_vals = np.zeros((z.shape[0] + 1, self._P_k.shape[0], self._P_k.shape[1]))
        P_k_1_vals = np.zeros((z.shape[0], self._P_k.shape[0], self._P_k.shape[1]))

        x_k_vals[0] = self._x_k
        P_k_vals[0] = self._P_k

        for i in range(1, z.shape[0] + 1):
            x_k_vals[i], x_k_1_vals[i - 1], P_k_vals[i], P_k_1_vals[i - 1] = self.step(u, z[i - 1])
        # print(x_k_vals)
        # print(x_k_vals)
        return (x_k_vals, x_k_1_vals, P_k_vals, P_k_1_vals)

    def smooth(self, u, z):
        x_k_vals, x_k_1_vals, P_k_vals, P_k_1_vals = self._filter_offline(u, z)

        smoothed_x_k = np.zeros((z.shape[0], self._x_k.shape[0]))
        smoothed_P_k = np.zeros((z.shape[0], self._P_k.shape[0], self._P_k.shape[1]))

        smoothed_x_k[-1] = x_k_vals[-1]
        smoothed_P_k[-1] = P_k_vals[-1]

        for k in range(z.shape[0] - 2, -1, -1):
            # on first pass k = K - 2 i.e. second to last value in array
            # L_K-2 = = P_K-2 @ A^T @ P_K+1[K-2] (prediciton of K-1 from K-2
            L_k = P_k_vals[k] @ T(self._A) @ I(P_k_1_vals[k])

            smoothed_x_k[k] = x_k_vals[k] + L_k @ (smoothed_x_k[k+1] - x_k_1_vals[k])

            smoothed_P_k[k] = P_k_vals[k] + L_k @ (smoothed_P_k[k+1] - P_k_1_vals[k]) @ T(L_k)

        return (x_k_vals, smoothed_x_k)








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
    a0y = -scipy.constants.g


    n = 100 # number of data samples
    t0 = 0 # start time
    tf = (-v0y - np.sqrt(v0y ** 2 - 4 * a0y/2 * y0)) / a0y # end time
    t = np.linspace(t0, tf, n+1) # time  vector
    dt = t[1]-t[0]

    # ignore control vector for now
    x = x0 + t * v0x + 0.5 * a0x * t ** 2
    y = y0 + t * v0y + 0.5 * a0y * t ** 2
    vy = v0y + a0y * t

    # add noise
    sigma = 0.8
    y_noisy = y + np.random.normal(0, sigma, n + 1)
    x_noisy = x + np.random.normal(0, sigma, n + 1)
    vx_noisy = v0x + np.random.normal(0, sigma, n + 1)
    vy_noisy = vy + np.random.normal(0, sigma, n + 1)
    ax_noisy = a0x + np.random.normal(0, sigma, n + 1)
    ay_noisy = a0y + np.random.normal(0, sigma, n + 1)
    print(ay_noisy)


    # attempt Kalman Filter
    initial_x=np.array([x0, v0x, a0x, y0, v0y, a0y], dtype='float')
    print('initial_x:', initial_x.shape)

    A = np.array([[1, dt, 0.5 * dt ** 2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, 0.5 * dt ** 2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]], dtype='float')
    print('A:', A.shape)

    H = np.identity(initial_x.shape[0])
    print('H:', H)

    B = np.zeros((initial_x.shape[0], initial_x.shape[0]))
    # B[2, 2] = 1
    # B[3, 3] = 1
    print('B:', B)

    P0 = np.identity(initial_x.shape[0])
    print('P0:', P0)

    Q = np.zeros((initial_x.shape[0], initial_x.shape[0]))
    print('Q:', Q)

    R = np.identity(initial_x.shape[0]) * sigma
    print('R:', R)

    # ignore control vector for now
    u = np.zeros(initial_x.shape[0]) # np.array([0, 0, a0y / 2 * dt ** 2, a0y * dt])

    initial_z = np.array([x_noisy[0], vx_noisy[0], ax_noisy[0], y_noisy[0], vy_noisy[0], ay_noisy[0]])
    print(initial_z.shape)

    kf = LinearKalmanFilter(A, B, P0, Q, R, H, initial_x, initial_z)

    # state_pred = np.empty((n + 1, 4))
    # for i in range(0, len(x)):
    #     z =[x_noisy[i], v0x + np.random.normal(0, sigma), y_noisy[i], v0y + np.random.normal(0, sigma)]
    #     print(np.array(z, dtype='float').shape)
    #     #print(z)
    #     kf.step(u, np.array(z, dtype='float'))
    #     state_pred[i] = kf.get_current_state()

    z = np.transpose(np.array([x_noisy, vx_noisy, ax_noisy, y_noisy, vy_noisy, ay_noisy]))
    z = z[1:,]
    print(z.shape)
    print(u.shape)

    state_pred, smoothed_state_pred = kf.smooth(u, z)
    print(state_pred[:, 0])
    plt.close()
    plt.figure()
    plt.scatter(x_noisy, y_noisy, label="Measured")
    plt.plot(x, y, '-g', label="True")
    plt.plot(state_pred[:, 0], state_pred[:, 3], '--r', label="Kalman")
    plt.plot(smoothed_state_pred[:, 0], smoothed_state_pred[:, 3], ':m', label="Kalman Smoothed")

    # print([min(min(x_noisy), min(x)) - 0.5, max(max(x_noisy), max(x)) + 0.5, min(min(y_noisy), min(y)) - 0.5, max(max(y_noisy), max(y)) + 0.5])
    plt.axis([min(min(x_noisy), min(x)) - 0.5, max(max(x_noisy), max(x)) + 0.5, min(min(y_noisy), min(y)) - 0.5, max(max(y_noisy), max(y)) + 0.5])
    plt.title('True vs Measured Position of a Projectile with Gaussian noise (\sigma=0.5)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
