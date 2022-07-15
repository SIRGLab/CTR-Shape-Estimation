import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Tube import Tube
from CTR_Observer import CTR_Observer, solve_in_time


if __name__ == "__main__":

    # Tube parameters:
    E = [3.7258e+10, 6.35704e+10]
    G = [3.43928e+10, 5.36801e+10]
    Ux = [14., 2.8]

    l = [0.5050, 0.251]  # length
    l_k = [0.1454, 0.1471]  # length curved

    # length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature, y_curvature
    tube1 = Tube(l[0], l_k[0], 2 * 0.35e-3, 2 * 0.55e-3, E[0], G[0], Ux[0], 0) # inner tube
    tube2 = Tube(l[1], l_k[1], 2 * 0.7e-3, 2 * 0.9e-3, E[1], G[1], Ux[1], 0) # outer tube
    tube3 = Tube(0, 0, 0, 0, 0, 0, 0, 0) # only using 2 tubes

    # Joint positions
    q = np.array([-0.28, -0.14, 0, 0, 0, 0])
    # force on robot tip along x, y, and z direction
    f = np.array([0., 0., 0.]).reshape(3, 1)
    # initial curvature
    u_init_0 = np.array([0., 0., 0., 0., 0.])

    # Initialize model
    CTR = CTR_Observer(tube1, tube2, tube3, f, q, 0.01)

    # Tip position values from an EM sensor
    EM = np.array([[0.0038], [-0.1129], [0.1399]])

    # Get CTR shape using bvp solving method as baseline
    u_init_bvp = CTR.minimize(u_init_0)
    CTR.ode_solver(u_init_bvp)
    shape_base = (CTR.r).T

    # Run observer
    u_0, errors = solve_in_time(CTR, u_0=u_init_0, tip_pos=EM, stop=1e-4, iter=20, step=1e-3, Q=30000, V=1480)
    shape_obs = (CTR.r).T

    # Plotting results
    fig_1 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(shape_base[0], shape_base[1], shape_base[2], '-r', label='Baseline Model')
    ax.scatter(EM[0,0], EM[1,0], EM[2,0], '-g',
           label='Sensor point')
    ax.plot(shape_obs[0], shape_obs[1], shape_obs[2], '-b', label='Observer')
    ax.auto_scale_xyz([np.amin(CTR.r[:, 0]) -0.05, np.amax(CTR.r[:, 0]) +0.05],
                      [np.amin(CTR.r[:, 1]) -0.05, np.amax(CTR.r[:, 1]) +0.05],
                      [np.amin(CTR.r[:, 2]) -0.05, np.amax(CTR.r[:, 2]) +0.05])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.grid(True)
    ax.legend()

    fig_2 = plt.figure()
    t = list(range(len(errors)))
    plt.plot(t, errors[:], label='Error at the tip, m')
    plt.grid(True)
    plt.title("Observer error")
    plt.xlabel("Iterations")
    plt.ylabel("Error at the tip, m")
    plt.legend()

    print("Observer error: %f" % errors[-1])
    print("Baseline error: %f" % np.linalg.norm(shape_base.T[-1].reshape(3,1) - EM))

    plt.show()