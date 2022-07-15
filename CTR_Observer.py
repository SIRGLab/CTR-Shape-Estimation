from Segment import Segment
import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
import scipy.io


e3 = np.array([0., 0., 1.]).reshape(3, 1)

# du^/du, 3x3x3 array
du_hat_du = np.array([[[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]],

                      [[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]],

                      [[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]]])


# skew-symmetric cross product matrix of vector v
def hat(v):
    return np.array([[0.0, -v[2, 0], v[1, 0]],
                     [v[2, 0], 0.0, -v[0, 0]],
                     [-v[1, 0], v[0, 0], 0.0]])

def R_theta(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def dR_theta(theta):
    return np.array([[-np.sin(theta), -np.cos(theta), 0],
                     [np.cos(theta), -np.sin(theta), 0],
                     [0, 0, 0]])


# Solves for curvature u_0 in time domain
# Returns optimised u_0 and a list of errors
def solve_in_time(CTR, u_0, tip_pos, stop=1e-4, iter=100, step=1e-3, Q=30000, V=1480):
    # Parameters:
    # CTR - instance of CTR_Observer
    # u_0 - initial guess for curvature u_0, e.g. np.array([[0],[0],[0]]), vector of of shape (3,1)
    # tip_pos - tip position measurement from a sensor, vector of of shape (3,1)
    # stop - stopping criteria
    # iter - max number of iterations
    # step - observer step size
    # Q and V are weights of Riccati eq.

    Q = np.eye(5) * Q
    V = np.eye(3) * V
    P = np.eye(5)
    errors = []

    for t in range(0, iter):

        CTR.reset()

        # Solve the equations in the spatial domain:
        CTR.ode_solver(u_0)

        # Get C and r at the end
        C = CTR.Cs[-1].reshape(3, 5)
        r = CTR.r[-1].reshape(3, 1)

        # Update time-dependent variables P and u_0
        dP = -P @ C.T @ V @ C @ P + Q
        P = P + step * dP

        # Update u_0 based on the error
        error = r - tip_pos
        du_0 = - P @ C.T @ V @ error
        u_0 = u_0 + step * du_0.reshape(5)

        errors.append(np.linalg.norm(error))
        if np.abs((step * np.linalg.norm(du_0)) / np.linalg.norm(u_0)) < stop:
            break

    return u_0, errors


class CTR_Observer:
    def __init__(self, tube1, tube2, tube3, f, q, Tol):
        self.accuracy = Tol
        self.eps = 1.e-4
        self.tube1, self.tube2, self.tube3, self.q = tube1, tube2, tube3, q.astype(float)
        # position of tubes' base from template (i.e., s=0)
        self.beta = self.q[0:3]
        self.f = f.astype(float)
        self.segment = Segment(self.tube1, self.tube2, self.tube3, self.beta)
        self.span = np.append([0], self.segment.S)

        # Initial values:
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # initial position of robot
        self.theta_1_0 = self.q[3]  # initial twist angle for tube 1
        self.R_0 = R_theta(self.theta_1_0).reshape(9, 1)  # initial rotation matrix
        self.theta_0 = q[3:].reshape(3, 1) - self.theta_1_0  # initial twist angle for all tubes
        # Initial values for Z, Gamma, C, D
        self.Z_0 = np.zeros((3, 5)).reshape(15,1)
        self.gamma_1_0 = np.hstack((np.eye(3), np.zeros((3, 2)))).reshape(15,1)
        self.gamma_2z_0 = np.array([0, 0, 0, 1, 0]).reshape(5,1)
        self.gamma_3z_0 = np.array([0, 0, 0, 0, 1]).reshape(5,1)
        self.C_0 = np.zeros((3, 5)).reshape(15,1)
        self.D_0 = np.zeros((3, 3, 5)).reshape(45, 1)

        self.Length = np.empty(0)
        self.r = np.empty((0, 3))
        self.u_z = np.empty((0, 3))
        self.theta = np.empty((0, 3))
        self.gammas = np.empty((0, 15))
        self.Cs = np.empty((0, 15))
        self.Zs = np.empty((0, 15))
        self.u1 = np.empty((0, 3))


    def reset(self):
        self.beta = self.q[0:3]
        self.segment = Segment(self.tube1, self.tube2, self.tube3, self.beta)
        self.span = np.append([0], self.segment.S)

        # Initial values:
        self.r_0 = np.array([0, 0, 0]).reshape(3, 1)  # initial position of robot
        self.theta_1_0 = self.q[3]  # initial twist angle for tube 1
        self.R_0 = R_theta(self.theta_1_0).reshape(9, 1)  # initial rotation matrix
        self.theta_0 = self.q[3:].reshape(3, 1) - self.theta_1_0  # initial twist angle for all tubes
        # Initial values for Z, Gamma, C, D
        self.Z_0 = np.zeros((3, 5)).reshape(15, 1)
        self.gamma_1_0 = np.hstack((np.eye(3), np.zeros((3, 2)))).reshape(15, 1)
        self.gamma_2z_0 = np.array([0, 0, 0, 1, 0]).reshape(5, 1)
        self.gamma_3z_0 = np.array([0, 0, 0, 0, 1]).reshape(5, 1)
        self.C_0 = np.zeros((3, 5)).reshape(15, 1)
        self.D_0 = np.zeros((3, 3, 5)).reshape(45, 1)

        self.Length = np.empty(0)
        self.r = np.empty((0, 3))
        self.u_z = np.empty((0, 3))
        self.theta = np.empty((0, 3))
        self.gammas = np.empty((0, 15))
        self.Cs = np.empty((0, 15))
        self.Zs = np.empty((0, 15))
        self.u1 = np.empty((0, 3))


    # A system of differential observer equations for a CTR with 3 tubes
    # Variables follow the same notations as the paper
    def ode_eq(self, s, y, ux_0, uy_0, ei, gj, f):
            # Parameter y is a vector of size (120,1):
            # [0:3] - u^1 (Curvature vector of the first tube)
            # [3] - u^2_z (twist curvature of the second tube)
            # [4] - u^3_z (twist curvature of the third tube)
            # [5:8] - theta_1, theta_2, theta_3 (twist angles)
            # [8:11] - r^1 (position of the first tube)
            # [11:20] - R^1 (orientation of the first tube)
            # [20:35] - Z^1, Z^2, Z^3,
            # [35:50] - gamma^1,
            # [50:55] - gamma^2_z (last row of gamma^2)
            # [55:60] - gamma^3_z (last row of gamma^3)
            # [60:75] - C
            # [75:120] - D

            dydt = np.empty([120, 1])

            theta1 = y[5]
            theta2 = y[6]
            theta3 = y[7]

            u1 = np.array(y[0:3]).reshape(3, 1)

            e3 = np.array([0., 0., 1.]).reshape(3, 1)
            u2_xy = R_theta(theta2).transpose() @ u1 + dydt[6] * e3  # Vector of curvature of tube 2
            u3_xy = R_theta(theta3).transpose() @ u1 + dydt[7] * e3

            u2 = np.vstack((u2_xy[:2], y[3]))
            u3 = np.vstack((u3_xy[:2], y[4]))

            u = np.array([y[0], y[1], y[2], u2[0, 0], u2[1, 0], y[3], u3[0, 0], u3[1, 0], y[4]])

            r = np.array([[y[8]], [y[9]], [y[10]]])

            R = np.array(
                [[y[11], y[12], y[13]], [y[14], y[15], y[16]], [y[17], y[18], y[19]]])  # rotation matrix of 1st tube

            K = [np.diag(np.array([ei[i], ei[i], gj[i]])) for i in range(len(ei))]

            u1_end = np.array([ux_0[0], uy_0[0], 0]).reshape(3, 1)
            u2_end = np.array([ux_0[1], uy_0[1], 0]).reshape(3, 1)
            u3_end = np.array([ux_0[2], uy_0[2], 0]).reshape(3, 1)
            us_end = np.array([u1_end, u2_end, u3_end]).reshape(3, 3, 1)

            dtheta1 = y[2] - y[2]
            dtheta2 = y[3] - y[2] if gj[1] != 0 else 0
            dtheta3 = y[4] - y[2] if gj[2] != 0 else 0

            u1 = y[0:3].reshape(3,1)
            u2 = np.vstack((u2[:2], y[3]))
            u3 = np.vstack((u3[:2], y[4]))
            us = np.array([u1, u2, u3]).reshape(3, 3, 1)

            # du_x and du_y of the first tube
            K_inv_sum = np.diag(np.array([1 / np.sum(ei), 1 / np.sum(ei), 1 / np.sum(gj)]))

            du1_xy = -K_inv_sum @ (
                      R_theta(theta1) @ (K[0] @ (dtheta1 * dR_theta(theta1).T @ u1 - 0) + hat(u1) @ K[0] @ (u1 - u1_end))
                    + R_theta(theta2) @ (K[1] @ (dtheta2 * dR_theta(theta2).T @ u1 - 0) + hat(u2) @ K[1] @ (u2 - u2_end))
                    + R_theta(theta3) @ (K[2] @ (dtheta3 * dR_theta(theta3).T @ u1 - 0) + hat(u3) @ K[2] @ (u3 - u3_end))) \
                    - K_inv_sum @ (hat(e3) @ R.transpose() @ f)

            dydt[0] = du1_xy[0, 0]
            dydt[1] = du1_xy[1, 0]

            # estimating twist curvature and twist angles
            for i in np.argwhere(gj != 0):
                dydt[2 + i] = ((ei[i]) / (gj[i])) * (u[i * 3] * uy_0[i] - u[i * 3 + 1] * ux_0[i])  # ui_z
                dydt[5 + i] = y[2 + i] - y[2]  # alpha_i
            for i in np.argwhere(gj == 0):
                dydt[2 + i] = 0
                dydt[5 + i] = 0

            # estimating R and r
            u_hat = np.array([[0, -y[2], y[1]], [y[2], 0, -y[0]], [-y[1], y[0], 0]])
            dr = R @ e3
            dR = (R @ u_hat).ravel()

            dydt[8] = dr[0, 0]
            dydt[9] = dr[1, 0]
            dydt[10] = dr[2, 0]

            for k in range(3, 12):
                dydt[8 + k] = dR[k - 3]


            # Observer eqs

            Z = np.array(y[20:35]).reshape(3, 5) # [dtheta_1/du(0,t), dtheta_2/du(0,t), dtheta_3/du(0,t)]
            gamma_1 = np.array(y[35:50]).reshape(3, 5) # du_1/du(0,t)
            gamma_2_z = np.array(y[50:55]).reshape(1, 5) # du_2_z/du(0,t)
            gamma_3_z = np.array(y[55:60]).reshape(1, 5) # du_3_z/du(0,t)
            C = np.array(y[60:75]).reshape(3, 5)
            D = np.array(y[75:120]).reshape(3, 3, 5)


            # Find dR_theta_i/du(0,t), i=0,1,2
            T1 = np.zeros((3,3,3,5)) # 3 arrays (for 3 tubes) of size 3x3x5
            thetas = np.array([theta1, theta2, theta3]).reshape(3,1)
            for i in range(3):
                T1[i, 0, 0, :] = np.array(-np.sin(thetas[i]) * Z[i])
                T1[i, 0, 1, :] = np.array(-np.cos(thetas[i]) * Z[i])
                T1[i, 1, 0, :] = np.array(np.cos(thetas[i]) * Z[i])
                T1[i, 1, 1, :] = np.array(-np.sin(thetas[i]) * Z[i])

            # Find du_i/du(0,t), i=0,1,2
            G = np.zeros((3, 3, 5))  # 3 arrays (for 3 tubes) of size 3x5
            G[0, :, :] = gamma_1
            # For tubes 2 and 3, find xy
            T1_T = T1.transpose(0, 2, 1, 3) # transpose axes 1 and 2 for dR_theta_i_T/du(0,t)
            gamma_2_xy = (u1.T @ T1_T[1]).reshape(3, 5) + R_theta(theta2).T @ gamma_1
            gamma_3_xy = (u1.T @ T1_T[2]).reshape(3, 5) + R_theta(theta3).T @ gamma_1
            G[1, :, :] = np.vstack((gamma_2_xy[:2, :], gamma_2_z))
            G[2, :, :] = np.vstack((gamma_3_xy[:2, :], gamma_3_z))

            # Define dZ
            dZ = G[:, 2, :] - G[0, 2, :]  # dtheta' = duz_i - duz_1
            for i in np.argwhere(gj == 0):
                dZ[i, :] = 0

            # Find d(theta_i * dR_theta_i^T/dtheta_i * u1)/d(0,t), i=0,1,2
            T2 = np.zeros((3, 3, 5))  # 3 arrays (for 3 tubes) of size 3x5
            # Define T4 = d(dR_theta_i/dtheta_i)^T/d(0,t), which is 3x3x5 matrix
            T4 = np.zeros((3, 3, 3, 5))  # 3 arrays (for 3 tubes) of size 3x3x5
            dthetas = np.array([dtheta1, dtheta2, dtheta3]).reshape(3, 1)
            for i in range(3):
                T4[i, 0, 0, :] = -np.cos(thetas[i,0]) * Z[i]
                T4[i, 0, 1, :] = -np.sin(thetas[i,0]) * Z[i]
                T4[i, 1, 0, :] = np.sin(thetas[i,0]) * Z[i]
                T4[i, 1, 1, :] = -np.cos(thetas[i,0]) * Z[i]
                T2[i, :, :] = (dR_theta(thetas[i,0]).T @ u1) @ dZ[i].reshape(1,5) + (dthetas[i,0] * u1.T @ T4[i]).reshape(3,5) + dthetas[i, 0] * (dR_theta(thetas[i,0]).T @ gamma_1)

            # Find du_i_hat/du(0,t), i=0,1,2
            T3 = np.zeros((3, 3, 3, 5))  # 3 arrays (for 3 tubes) of size 3x3x5
            for i in range(3):
                T3[i] = (du_hat_du @ G[i]).reshape(3,3,5)


            # Derivatives

            dgamma_1_xy_ = np.zeros((3, 5))
            for j in np.argwhere(gj != 0):
                i = j[0]
                dgamma_1_xy_ += np.array(
                    ((K[i] @ (dthetas[i, 0] * dR_theta(thetas[i, 0]).T @ u1) + hat(us[i]) @ K[i] @ (us[i] - us_end[i])).T @ T1[i]).reshape(3, 5)
                    + R_theta(thetas[i, 0]) @ (K[i] @ T2[i] + ((K[i] @ (us[i] - us_end[i])).T @ T3[i]).reshape(3, 5) + hat(us[i]) @ K[i] @ G[i]))

            dgamma_1_xy = - K_inv_sum @ dgamma_1_xy_ - K_inv_sum @ hat(e3) @ (f.T @ D.transpose(1,0,2)).reshape(3,5)

            dgamma_1_z = ei[0]/gj[0] * (u1_end[1] * gamma_1[0, :] - u1_end[0] * gamma_1[1, :]) if gj[0] != 0 else np.zeros((1, 5))
            dgamma_1 = np.vstack((dgamma_1_xy[:2, :], dgamma_1_z))

            dgamma_2_z = ei[1]/gj[1] * (u2_end[1] * G[1, 0, :] - u2_end[0] * G[1, 1, :]) if gj[1] != 0 else np.zeros((1, 5))
            dgamma_3_z = ei[2]/gj[2] * (u3_end[1] * G[2, 0, :] - u3_end[0] * G[2, 1, :]) if gj[2] != 0 else np.zeros((1, 5))

            dC = (e3.T @ D).reshape(3, 5)
            dD = (hat(u1)).T @ D + (R @ (T3[0]).transpose(1, 0, 2)).transpose(1, 0, 2)

            dydt[20:35, :] = dZ.reshape(15, 1)
            dydt[35:50, :] = dgamma_1.reshape(15,1)
            dydt[50:55, :] = dgamma_2_z.reshape(5, 1)
            dydt[55:60, :] = dgamma_3_z.reshape(5, 1)
            dydt[60:75, :] = dC.reshape(15, 1)
            dydt[75:120, :] = dD.reshape(45, 1)

            return dydt.ravel()

    def ode_solver(self, u_init):
        # or not
        u1_xy_0 = np.array([[0.0], [0.0]])
        u1_xy_0[0, 0] = u_init[0]
        u1_xy_0[1, 0] = u_init[1]
        uz_0 = u_init[2:].reshape(3, 1)

        # reset initial parameters for ode solver
        self.reset()
        for seg in range(0, len(self.segment.S)):
            # y_0 is a vector of initial conditions for ode_eq
            y_0 = np.vstack((u1_xy_0, uz_0, self.theta_0, self.r_0, self.R_0, self.Z_0, self.gamma_1_0, self.gamma_2z_0, self.gamma_3z_0, self.C_0, self.D_0)).ravel()
            s = solve_ivp(lambda s, y: self.ode_eq(s, y, self.segment.U_x[:, seg], self.segment.U_y[:, seg],
                                                   self.segment.EI[:, seg], self.segment.GJ[:, seg], self.f),
                          [self.span[seg], self.span[seg + 1]],
                          y_0, method='RK23', max_step=self.accuracy)
            self.Length = np.append(self.Length, s.t)
            ans = s.y.transpose()
            self.u1 = np.vstack((self.u1, ans[:, (0, 1, 2)]))
            self.u_z = np.vstack((self.u_z, ans[:, (2, 3, 4)]))
            self.theta = np.vstack((self.theta, ans[:, (5, 6, 7)]))
            self.r = np.vstack((self.r, ans[:, (8, 9, 10)]))
            self.gammas = np.vstack((self.gammas, ans[:, 35:50]))
            self.Cs = np.vstack((self.Cs, ans[:, 60:75]))
            self.Zs = np.vstack((self.Zs, ans[:, 20:35]))
            dtheta2 = ans[-1, 3] - ans[-1, 2]
            dtheta3 = ans[-1, 4] - ans[-1, 2]
            # new boundary conditions for next segment
            uz_0 = self.u_z[-1, :].reshape(3, 1)
            self.r_0 = self.r[-1, :].reshape(3, 1)
            self.R_0 = np.array(ans[-1, 11:20]).reshape(9, 1)
            self.theta_0 = self.theta[-1, :].reshape(3, 1)

            self.Z_0 = np.array(ans[-1, 20:35]).reshape(15, 1)
            self.gamma_1_0 = np.array(ans[-1, 35:50]).reshape(15, 1)
            self.gamma_2z_0 = np.array(ans[-1, 50:55]).reshape(5, 1)
            self.gamma_3z_0 = np.array(ans[-1, 55:60]).reshape(5, 1)
            self.C_0 = np.array(ans[-1, 60:75]).reshape(15, 1)
            self.D_0 = np.array(ans[-1, 75:120]).reshape(45, 1)

            u1 = ans[-1, (0, 1, 2)].reshape(3, 1)

            if seg < len(
                    self.segment.S) - 1:  # enforcing continuity of moment to estimate initial curvature for next
                # segment

                K1 = np.diag(np.array([self.segment.EI[0, seg], self.segment.EI[0, seg], self.segment.GJ[0, seg]]))
                K2 = np.diag(np.array([self.segment.EI[1, seg], self.segment.EI[1, seg], self.segment.GJ[1, seg]]))
                K3 = np.diag(np.array([self.segment.EI[2, seg], self.segment.EI[2, seg], self.segment.GJ[2, seg]]))
                U1 = np.array([self.segment.U_x[0, seg], self.segment.U_y[0, seg], 0]).reshape(3, 1)
                U2 = np.array([self.segment.U_x[1, seg], self.segment.U_y[1, seg], 0]).reshape(3, 1)
                U3 = np.array([self.segment.U_x[2, seg], self.segment.U_y[2, seg], 0]).reshape(3, 1)

                GJ = self.segment.GJ
                GJ[self.segment.EI[:, seg + 1] == 0] = 0
                K1_new = np.diag(
                    np.array([self.segment.EI[0, seg + 1], self.segment.EI[0, seg + 1], self.segment.GJ[0, seg + 1]]))
                K2_new = np.diag(
                    np.array([self.segment.EI[1, seg + 1], self.segment.EI[1, seg + 1], self.segment.GJ[1, seg + 1]]))
                K3_new = np.diag(
                    np.array([self.segment.EI[2, seg + 1], self.segment.EI[2, seg + 1], self.segment.GJ[2, seg + 1]]))
                U1_new = np.array([self.segment.U_x[0, seg + 1], self.segment.U_y[0, seg + 1], 0]).reshape(3, 1)
                U2_new = np.array([self.segment.U_x[1, seg + 1], self.segment.U_y[1, seg + 1], 0]).reshape(3, 1)
                U3_new = np.array([self.segment.U_x[2, seg + 1], self.segment.U_y[2, seg + 1], 0]).reshape(3, 1)

                R_theta2 = R_theta(self.theta_0[1, 0])
                R_theta3 = R_theta(self.theta_0[2, 0])
                u2 = R_theta2.transpose() @ u1 + dtheta2 * e3
                u3 = R_theta3.transpose() @ u1 + dtheta3 * e3
                K_inv_new = np.diag(
                    np.array(
                        [1 / (self.segment.EI[0, seg + 1] + self.segment.EI[1, seg + 1] + self.segment.EI[2, seg + 1]),
                         1 / (self.segment.EI[0, seg + 1] + self.segment.EI[1, seg + 1] + self.segment.EI[2, seg + 1]),
                         1 / (self.segment.GJ[0, seg + 1] + self.segment.GJ[1, seg + 1] + self.segment.GJ[
                             2, seg + 1])]))
                u1_new = K_inv_new @ (K1 @ (u1 - U1) + R_theta2 @ K2 @ (u2 - U2) + R_theta3 @ K3 @ (
                        u3 - U3) + K1_new @ U1_new + R_theta2 @ K2_new @ U2_new + R_theta3 @ K3_new @ U3_new
                                      - R_theta2 @ K2_new @ (dtheta2 * e3) - R_theta3 @ K3_new @ (
                                              dtheta3 * e3))
                u1_xy_0 = u1_new[0:2, 0].reshape(2, 1)

                # Estimate initial Z and gamma for the next segment
                Z = self.Z_0.reshape(3,5)
                gamma_1 = self.gamma_1_0.reshape(3,5)

                # Find dR_theta_i/du(0,t), i=0,1,2
                T1 = np.zeros((3, 3, 3, 5))  # 3 arrays (for 3 tubes) of size 3x3x5
                thetas = self.theta_0.reshape(3,1)
                for i in range(3):
                    T1[i, 0, 0, :] = np.array(-np.sin(thetas[i, 0]) * Z[i])
                    T1[i, 0, 1, :] = np.array(-np.cos(thetas[i, 0]) * Z[i])
                    T1[i, 1, 0, :] = np.array(np.cos(thetas[i, 0]) * Z[i])
                    T1[i, 1, 1, :] = np.array(-np.sin(thetas[i, 0]) * Z[i])

                # Find du_i/du(0,t), i=0,1,2
                G = np.zeros((3, 3, 5))  # 3 arrays (for 3 tubes) of size 3x5
                G[0, :, :] = gamma_1
                # For tubes 2 and 3, find xy
                T1_T = T1.transpose(0, 2, 1, 3)  # transpose axes 1 and 2 for dR_theta_i_T/du(0,t)
                gamma_2_xy = (u1.T @ T1_T[1]).reshape(3, 5) + R_theta2.T @ gamma_1
                gamma_3_xy = (u1.T @ T1_T[2]).reshape(3, 5) + R_theta3.T @ gamma_1
                G[1, :, :] = np.vstack((gamma_2_xy[:2, :], self.gamma_2z_0.reshape(1,5)))
                G[2, :, :] = np.vstack((gamma_3_xy[:2, :], self.gamma_3z_0.reshape(1,5)))

                # Define dZ
                dZ = G[:, 2, :] - G[0, 2, :]  # dtheta' = duz_i - duz_1
                for i in np.argwhere(self.segment.GJ[:, seg + 1] == 0):
                    dZ[i, :] = 0

                gamma_new = K_inv_new @ (K1 @ gamma_1
                                         + ((K2 @ (u2 - U2)).T @ T1[1]).reshape(3,5) + R_theta2 @ K2 @ G[1]
                                         + ((K3 @ (u3 - U3)).T @ T1[2]).reshape(3,5) + R_theta3 @ K3 @ G[2]
                                         + ((K2_new @ U2_new).T @ T1[1]).reshape(3,5)
                                         + ((K3_new @ U3_new).T @ T1[2]).reshape(3,5)
                                         - ((K2_new @ (dtheta2 * e3)).T @ T1[1]).reshape(3,5) - R_theta2 @ K2_new @ (dZ[1,:] * e3)
                                         - ((K3_new @ (dtheta3 * e3)).T @ T1[2]).reshape(3,5) - R_theta3 @ K3_new @ (dZ[2,:] * e3))

                gamma_new[2,:] = np.copy(gamma_1[2,:])
                self.gamma_1_0 = gamma_new.reshape(15,1)

        Cost = np.array([u1[0, 0] - self.segment.U_x[0, -1], u1[1, 0] - self.segment.U_y[0, -1], u1[2, 0], 0.0,
                         0.0])  # cost function for bpv solver includes 5 values: 3 twist curvature for tip of the 3
        # tubes and end curvature of the tip of the robot

        # finding twist curvature at the tip of the tubes
        d_tip = np.array([self.tube1.L, self.tube2.L, self.tube3.L]) + self.beta
        for i in range(1, 3):
            index = np.argmin(abs(self.Length - d_tip[i]))
            Cost[i + 2] = self.u_z[index, i]
        return Cost

    # Solving the BVP problem using built-in scipy minimize module
    def minimize(self, u_init):
        u0 = u_init
        u0[0] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                (self.segment.EI[0, 0] * self.segment.U_x[0, 0] +
                 self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.cos(- self.theta_0[1, 0]) +
                 self.segment.EI[1, 0] *
                 self.segment.U_y[1, 0] * np.sin(- self.theta_0[1, 0]) +
                 self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.cos(- self.theta_0[2, 0]) +
                 self.segment.EI[2, 0] *
                 self.segment.U_y[2, 0] * np.sin(- self.theta_0[2, 0]))
        u0[1] = (1 / (self.segment.EI[0, 0] + self.segment.EI[1, 0] + self.segment.EI[2, 0])) * \
                (self.segment.EI[0, 0] * self.segment.U_y[0, 0] +
                 -self.segment.EI[1, 0] * self.segment.U_x[1, 0] * np.sin(- self.theta_0[1, 0]) +
                 self.segment.EI[1, 0] *
                 self.segment.U_y[1, 0] * np.cos(- self.theta_0[1, 0]) +
                 -self.segment.EI[2, 0] * self.segment.U_x[2, 0] * np.sin(- self.theta_0[2, 0]) +
                 self.segment.EI[2, 0] *
                 self.segment.U_y[2, 0] * np.cos(-self.theta_0[2, 0]))
        res = optimize.anderson(self.ode_solver, u0, f_tol=1e-3)
        return res