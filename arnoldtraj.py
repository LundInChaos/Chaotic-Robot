import scipy.integrate as integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random as ran
from collections import deque
from numpy import linalg as LA
import math
import argparse
import sys

'''
Script for generating 3d trajectory for arnold flow and lyapunov exponent
'''

def arnold(dxs, t):
    dx1 = A * np.sin(dxs[2]) + C * np.cos(dxs[1])
    dx2 = B * np.sin(dxs[0]) + A * np.cos(dxs[2])
    dx3 = C * np.sin(dxs[1]) + B * np.cos(dxs[0])

    return np.asarray([dx1, dx2, dx3])

def check_input(s: str):
    if s.lower() == 'y':
        return True
    elif s.lower() == 'n':
        return False
    else:
        return 'Invalid input!'
# ===============================================================
# Set traj = True: generate traj of the arnold flow
# Set lya_exp = True: calculate the largest lyapunov exponent
# Set poin = True: generate poincare sectionof the arnold flow
# ===============================================================
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# subparser = parser.add_subparsers(dest='command')


parser.add_argument('--traj', help='Generate trajectory of arnold flow.', action="store_true")
parser.add_argument('--lyapunov', help='Calculate largest lyapunov exponent for arnold flow.', action="store_true")
parser.add_argument('--poincare', help='Generate poincare sections.', action="store_true")

args = parser.parse_args()

traj = args.traj
lya_exp = args.lyapunov
poin = args.poincare

if not traj and not lya_exp and not poin:
    parser.print_help()
    sys.exit(0)

while True:

    t0   = 0
    tEnd = 5000
    t    = np.linspace( t0, tEnd, num = tEnd+1 )

    x0   = np.asarray( [4, 3.5, 0] )
    x1   = np.asarray( [2.1, -3, 0] )

    fig = plt.figure()

    # ------------Lyapunov exponent---------------
    if lya_exp:
        A = 0.5
        B = 0.25
        C = 0.25
        epsilon = 1e-8
        tt = []
        exp = []
        x00 = np.asarray([4 + epsilon, 3.5, 3])
        x_1 = x0[0]
        x_2 = x0[1]
        x_3 = x0[2]
        jacob_init = np.array([[0, -C * np.sin(x_2), A * np.cos(x_3)], [B * np.cos(x_1), 0, -A * np.sin(x_3)], [-B * np.sin(x_1), C * np.cos(x_2), 0]])
        m = jacob_init
        # h1 = math.log(LA.norm(jacob_init, axis=0)[0])
        # h2 = math.log(LA.norm(jacob_init, axis=0)[1])
        # h3 = math.log(LA.norm(jacob_init, axis=0)[2])
        for i in range(1):
            f = integrate.odeint( arnold, x0, t )
            fe = integrate.odeint( arnold, x00, t )
            for j in range(len(f)):
                x_1 = f[j, 0]
                # print(f[j, 0])
                x_2 = f[j, 1]
                x_3 = f[j, 2]
                dx_1 = fe[j, 0]
                dx_2 = fe[j, 1]
                dx_3 = fe[j, 2]
                d1 = dx_1 - x_1
                d2 = dx_2 - x_2
                d3 = dx_3 - x_3
                dN = math.sqrt(d1 ** 2 + d2 ** 2 + d3 ** 2)
                e1 = math.log(dN / epsilon) / (j + 1)
                exp.append(e1)
                tt.append((j+1))
                # px = A * C * math.cos(x_2) * math.sin(x_3) + B * C * math.sin(x_2) * math.cos(x_1) + A * B * math.cos(x_3) * math.sin(x_1)
                # p = A * B * C * (math.cos(x_1) * math.cos(x_2) * math.cos(x_3) - math.sin(x_1) * math.sin(x_2) * math.sin(x_3))
                # h1, h2, h3 = np.roots([1, 0, px, p])
                '''
                ############# Jacobian method #############
                jacob = np.array([[0, -C * np.sin(x_2), A * np.cos(x_3)], [B * np.cos(x_1), 0, -A * np.sin(x_3)],
                      [-B * np.sin(x_1), C * np.cos(x_2), 0]])
                m = np.matmul(m, jacob)
                # print(m)
                # try:
                #     h1 = math.log(LA.norm(m, axis=0)[0])
                #     h2 = math.log(LA.norm(m, axis=0)[1])
                #     h3 = math.log(LA.norm(m, axis=0)[2])
                # except:
                #     continue
                # print()
                h1 = LA.eig(m)[0][0]
                h2 = LA.eig(m)[0][1]
                h3 = LA.eig(m)[0][2]
                # print(h1, h2, h3)
                h1 = math.log(abs(h1))
                h2 = math.log(abs(h2))
                h3 = math.log(abs(h3))
                try:
                    exp.append([h1/(j+1), h2/(j+1), h3/(j+1)])
                    tt.append(t[j])
                except:
                    continue
                '''
            exp = np.asarray(exp)
            print(exp[-1])
            # print(h1, h2, h3)
            # print(f[-1])
            plt.plot(tt, exp, label='$lambda_1$', color='blue', linewidth=0.8)
            # plt.plot(t, exp[:, 1], label='$lambda_2$', color='red', linewidth=0.8)
            # plt.plot(t, exp[:, 2], label='$lambda_3$', color='green', linewidth=0.8)
            axes = plt.gca()
            axes.set_xlabel('Iterations', fontsize='20')
            axes.set_ylabel('$\lambda$', fontsize='20')
            plt.show()
        break

    # ------------3D or 2D trajectory---------------
    elif traj:
        t0 = 0
        tEnd = 200
        t = np.linspace(t0, tEnd, num=tEnd + 1)
        chaos = check_input(input('Chaos? [Y/N]'))
        if isinstance(chaos, str):
            print(chaos)
            continue
        three_D = check_input(input('3D? [Y/N]'))
        if isinstance(three_D, str):
            print(three_D)
            continue
        A = 1
        B = 0.5
        if chaos:
            C = 0.5
        else:
            C = 0
        if three_D:
            ax = fig.gca(projection='3d')
            for i in range(20):
                # x = 1 + 0.11 * i
                y = -10 + 0.8 * i
                # z = 0 + 0.05 * i
                start_input = np.asarray([1, y, 0.5])
                g = []
                f = integrate.odeint( arnold, start_input, t )
                for j in range(len(f)):
                    if ( f[j, 1] >= -1 ) and ( f[j, 1] <= 20 ):
                        g.append([f[j, 0],f[j, 1],f[j, 2]])
                g = np.asarray(g)
                ax.plot(g[:, 0], g[:, 1], g[:, 2], linewidth=0.8, color='blue')
            #
            # axes = plt.gca()
            # axes.set_xlabel('Iterations', fontsize = '20')
            # axes.set_ylabel('$\lambda$', fontsize = '20')
            #
            ax.set_xlabel('$x_1$', fontsize = '20')
            ax.set_ylabel('$x_2$', fontsize = '20')
            ax.set_zlabel('$x_3$', fontsize = '20')
            plt.show()
        else:
            for i in range(20):
                # x = 1 + 0.11 * i
                y = -10 + 0.8 * i
                # z = 0 + 0.05 * i
                start_input = np.asarray([1, y, 0.5])
                f = integrate.odeint( arnold, start_input, t )
                plt.plot(f[:, 0], f[:, 2], linewidth=0.8, color='blue')
            axes = plt.gca()
            axes.set_xlabel('$x_1$', fontsize = '20')
            axes.set_ylabel('$x_3$', fontsize = '20')
            plt.show()
        break

    # ------------Poincare section---------------
    elif poin:
        t0 = 0
        tEnd = 10
        t = np.linspace(t0, tEnd, num=tEnd + 1)

        A = 1
        B = 0.5
        chaos = check_input(input('Chaos? [Y/N]'))
        if isinstance(chaos, str):
            print(chaos)
            continue
        if chaos:
            C = 0.5
        else:
            C = 0

        x0 = np.asarray([2, -3, 1])
        g = []


        def arnold_poin(dxs, t):
            dx1 = A * np.sin(dxs[1]) + C
            dx2 = B * np.cos(dxs[0])

            return np.asarray([dx1, dx2])

        for i in range(20):
            x = -10 + 1 * i
            for j in range(50):
                y = 0 + 0.5 * j
                x0 = np.asarray([x, y])
                f = integrate.odeint(arnold_poin, x0, t)
                plt.plot(f[:, 0], f[:, 1], linewidth=0.8, color='black')
                # plt.plot(f[:,0], f[:,2], linewidth = 0.8, color = 'blue')

        axes = plt.gca()
        axes.set_xlabel('$x_1$', fontsize='20')
        axes.set_ylabel('$x_3$', fontsize='20')

        plt.show()
        break
