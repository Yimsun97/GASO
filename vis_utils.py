import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mtick

from opt_utils import schaffer, squared


class AnimatedScatter:
    """散点动态图"""
    def __init__(self, func, lb, ub, points_list, n_grid=1000, file_name=None):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.ps_lst = points_list
        self.n_grid = n_grid
        self.scat = None

        # 设置图和轴
        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))

        # 设置动画函数
        repeat_flag = False if file_name else True
        self.ani = animation.FuncAnimation(self.fig, self.update,
                                           frames=len(self.ps_lst), interval=500,
                                           init_func=self.setup_plot, blit=True,
                                           repeat=repeat_flag)
        # 保存GIF
        if file_name:
            self.ani.save(file_name)

    def setup_plot(self):
        """初始化函数"""
        # 数据
        x_lim, y_lim = [*zip(self.lb, self.ub)]
        X, Y, Z = data_generate(self.func, self.lb, self.ub, self.n_grid)
        x, y = self.ps_lst[0].T
        # 初始化图
        self.ax.axis([*x_lim, *y_lim])
        self.ax.contourf(X, Y, Z, cmap='PuBu_r')
        self.scat = self.ax.scatter(x, y, c='red')
        return self.scat,

    def update(self, i):
        """更新散点图"""
        ii = i % len(self.ps_lst)
        data = self.ps_lst[ii]
        # 设置数据
        self.scat.set_offsets(data)

        return self.scat,


def data_generate(func, lb, ub, n_grid=1000):
    # 生成数据
    x_lim, y_lim = [*zip(lb, ub)]
    x = np.linspace(*x_lim, n_grid)
    y = np.linspace(*y_lim, n_grid)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y])
    return X, Y, Z


def vis_2d(func, lb, ub, points, n_grid=1000):
    # 生成数据
    x_lim, y_lim = [*zip(lb, ub)]
    X, Y, Z = data_generate(func, lb, ub, n_grid)
    # 画图
    fig = plt.figure()
    cs = plt.contourf(X, Y, Z, cmap='PuBu_r')
    plt.plot(points[:, 0], points[:, 1], 'r.')
    fig.colorbar(cs)
    plt.xlim(x_lim)
    plt.ylim(y_lim)


def vis_iter(history, legend):
    plt.figure()
    plt.plot(history, 'r', label=legend)
    plt.legend()
    plt.show()


def vis_iter_agg(eval_lst, history_lst, legend_lst, scale='linear'):
    plt.figure(figsize=(10, 7.5))
    plt.rcParams.update({'font.size': 18,
                         'font.family': 'Times New Roman'})
    for i in range(len(history_lst)):
        plt.plot(eval_lst[i], history_lst[i],
                 label=legend_lst[i], linewidth=3)
    plt.xlabel("Total Number of Evaluations")
    plt.ylabel("Optimal Value")
    plt.legend()

    if scale == 'linear':
        # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        if np.max(history_lst[0]) < 0:
            plt.ylim([-20, 0])

    if scale == 'log':
        plt.yscale(scale)

    plt.show()


if __name__ == '__main__':
    ps = np.random.randn(100, 2) * 100
    vis_2d(schaffer, lb=[-100, -100], ub=[100, 100], points=ps)
    psl = [np.random.randn(100, 2) * 100 for i in range(10)]
    a = AnimatedScatter(
        func=schaffer,
        lb=[-100, -100],
        ub=[100, 100],
        points_list=psl
    )
