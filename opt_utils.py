import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


def jong(x):
    numer = 100 * (x[0] ** 2.0 - x[1]) ** 2.0 + (1 - x[0]) ** 2.0
    denom = 1000
    opt_target = numer / denom
    return opt_target


def schaffer(x):
    fx = x[0] ** 2.0 + x[1] ** 2.0
    numer = (np.sin(np.sqrt(fx))) ** 2.0 - 0.5
    denom = (1 + 0.001 * fx) ** 2.0
    opt_target = 4.5 - numer / denom
    return opt_target


def schaffer_modified(x):
    fx = x[0] ** 2.0 + (x[1] - 100) ** 2.0
    numer = (np.sin(np.sqrt(fx))) ** 2.0 - 0.5
    denom = (1 + 0.001 * fx) ** 2.0
    opt_target = 4.5 - numer / denom
    return opt_target


def schaffer_nd(x, n=3):
    fx = sum([x[i] ** 2.0 + (x[i + 1]) ** 2.0
              for i in range(n - 1)])
    numer = (np.sin(np.sqrt(fx))) ** 2.0 - 0.5
    denom = (1 + 0.001 * fx) ** 2.0
    opt_target = 4.5 - numer / denom
    return opt_target


def squared(x):
    return -x[0] ** 2.0 - x[1] ** 2.0


def Rastrigin(x):
    A = 10
    B = x[0] ** 2.0 - A * np.cos(2 * np.pi * x[0])
    C = x[1] ** 2.0 - A * np.cos(2 * np.pi * x[1])
    return -(A * 2 + (B + C))


def Ackley(x):
    xx, yy = x[0], x[1]
    A = -20 * np.exp(-0.2 * np.sqrt(0.5 * (xx ** 2.0 + yy ** 2.0)))
    B = -np.exp(0.5 * (np.cos(2 * np.pi * xx) + np.cos(2 * np.pi * yy)))
    return -(A + B + np.exp(1) + 20)


def Rosenbrock(x):
    return -(100 * (x[1] - x[0] ** 2.0) ** 2.0 + (1 - x[0]) ** 2.0)


def Rosenbrock_nd(x, n=3):
    f = [100 * (x[i + 1] - x[i] ** 2.0) ** 2.0 + (1 - x[i]) ** 2.0
         for i in range(n - 1)]
    return -sum(f)


def Beale(x):
    A = (1.5 - x[0] + x[0] * x[1]) ** 2.0
    B = (2.25 - x[0] + x[0] * x[1] ** 2.0) ** 2.0
    C = (2.625 - x[0] + x[0] * x[1] ** 3.0) ** 2.0
    return -(A + B + C)


def Himmelblau(x):
    A = (x[0] ** 2.0 + x[1] - 11) ** 2.0
    B = (x[0] + x[1] ** 2.0 - 7) ** 2.0
    return -(A + B)


def Booth(x):
    A = (x[0] + 2 * x[1] - 7) ** 2.0
    B = (2 * x[0] + x[1] - 5) ** 2.0
    return -(A + B)


def Bukin(x):
    A = 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2.0))
    B = 0.01 * np.abs(x[0] + 10)
    return -(A + B)


def ode_loss_func(x, fig=False):
    SEED = 1024
    np.random.seed(SEED)

    def ode(p):
        rhs_func = lambda t, y: p[0] + p[1] * t
        rhs_func = lambda t, y: -p[0] * y + p[1] * np.sin(t) + 0.5
        _sol = solve_ivp(rhs_func, [0, 100], [2.0],
                         method='RK45', dense_output=True)
        return _sol

    x_true = [0.2, 0.5]
    num = 30
    sol_true = ode(x_true)
    tt = np.linspace(0, 100, num)
    yy = sol_true.sol(tt)  # + np.random.randn(1, num) * 0.07 * np.mean(sol_true.sol(tt))

    X, Y = x
    XX = X.ravel()
    YY = Y.ravel()

    rmse = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        rmse[i] = np.sum((yy - ode([XX[i], YY[i]]).sol(tt)) ** 2.0)

    if isinstance(x, list):
        rmse = rmse.reshape(X.shape[0], -1)

    if fig:
        t_ = np.linspace(0, 100, 500)
        plt.figure(figsize=(10, 6))
        plt.plot(tt, yy.ravel(), 'ro', label="Observed")
        plt.plot(t_, ode(x_true).sol(t_).ravel(), 'b--', alpha=0.5, label="True")
        plt.plot(t_, ode(x).sol(t_).ravel(), 'C0', alpha=0.5, label="Predicted")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("x2")
        plt.show()

    return -rmse


if __name__ == '__main__':
    ode_loss_func(np.array([0.2, 0.5]), fig=True)
