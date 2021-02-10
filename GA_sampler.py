import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from SALib.sample.latin import sample
from matplotlib import animation, rcParams

from opt_utils import *
from vis_utils import AnimatedScatter


class GASO:

    def __init__(self, func, n_dim, n_iter, n_inits,
                 lb, ub, pcr, pse, psp, n_samples,
                 p_epsilon=0.05, theta=0.5):
        self.func = func
        self.n_dim = n_dim
        self.n_iter = n_iter
        self.n_inits = n_inits
        self.lb = lb
        self.ub = ub
        self.pcr = pcr
        self.pa = pse
        self.psp = psp
        self.n_samples = n_samples
        self.p_epsilon = p_epsilon
        self.theta = theta

        self.interval = np.array(self.ub) - np.array(self.lb)
        self.history_best = []
        self.storage = {}
        self.snap = []
        self.history_fitness = np.zeros((0, self.n_dim + 1))
        self.eval_count = 0
        self.eval_count_list = []

    def mutation(self):
        pass

    def init_sampling(self):
        _init_samples = self.sampler(self.lb, self.ub, self.n_inits)
        # 评价适应度
        _y = self.fitness(_init_samples)
        ps_new = np.column_stack([_init_samples, _y])
        aps_idx = np.argsort(_y)[::-1][:int(self.n_inits * self.pa)]
        ips_idx = np.argsort(_y)[::-1][int(self.n_inits * self.pa):]
        # 初始化APs/FPs/IPs
        self.storage['APs'] = ps_new[aps_idx]
        self.storage['FPs'] = np.zeros((0, self.n_dim + 1))
        self.storage['IPs'] = ps_new[ips_idx]

    def fitness(self, _samples):
        _fitness = np.zeros((_samples.shape[0]))

        def fitness_eval():
            _fitness[i] = self.func(samp.T)
            _fit_cache = np.concatenate([samp, [_fitness[i]]])
            self.history_fitness = np.vstack([x for x in [self.history_fitness, _fit_cache]
                                              if x.size > 0])
            self.eval_count += 1

        for i, samp in enumerate(_samples):
            if len(self.history_fitness) > 1:
                hist_fit = self.history_fitness[:, :-1]
                if samp in hist_fit:
                    _fitness[i] = self.history_fitness[(samp == hist_fit).any(1), -1][0]
                else:
                    fitness_eval()
            else:
                fitness_eval()

        return _fitness

    def sampler(self, lb, ub, n_samples):
        problem = {
            'num_vars': self.n_dim,
            'bounds': [*zip(lb, ub)]
        }
        _samples = sample(problem, N=n_samples)
        return _samples

    def split_samples(self, _samples, _ap):
        # 评价适应度
        _y = self.fitness(_samples)
        ps_new = np.column_stack([_samples, _y])

        # 单个active point
        aps_single = _ap
        _ap_x = aps_single[:-1]
        _ap_y = aps_single[-1]

        # 确定新的APs/FPs/IPs
        if np.any(_y >= _ap_y):
            _aps = ps_new[np.argmax(_y)]
            _ips = ps_new[np.argsort(_y)[::-1][1:]]
            _fps = _ap
        else:
            _aps = _ap
            _ips = ps_new
            _fps = np.zeros((0, self.n_dim + 1))
        return _aps, _fps, _ips

    def update_samples(self):
        aps = self.storage['APs']
        aps_y = aps[:, -1]
        top_aps_idx = np.argsort(aps_y)[::-1][:int(aps.shape[0] * self.psp)]
        other_aps_idx = np.argsort(aps_y)[::-1][int(aps.shape[0] * self.psp):]
        top_aps = aps[top_aps_idx]
        top_aps_x = top_aps[:, :-1]

        aps_cache = []
        fps_cache = []
        ips_cache = []

        # 在前psp的active points邻域内采样
        for i in range(len(top_aps)):
            lb = top_aps_x[i] - np.ones_like(self.lb) * self.p_epsilon * self.interval
            ub = top_aps_x[i] + np.ones_like(self.lb) * self.p_epsilon * self.interval

            # 修正lb/ub以防超出界限
            lb = np.max(np.column_stack([np.array(self.lb), lb]), axis=1)
            ub = np.min(np.column_stack([np.array(self.ub), ub]), axis=1)

            _samples = self.sampler(lb, ub, self.n_samples)
            ap_sample, fp_sample, ip_sample = self.split_samples(_samples, top_aps[i])

            aps_cache.append(ap_sample)
            fps_cache.append(fp_sample)
            ips_cache.append(ip_sample)

        # 非顶级的active points不采样仍保留
        aps_cache.append(aps[other_aps_idx])

        self.storage['APs'] = np.vstack([x for x in aps_cache]) \
            if len(aps_cache) > 0 else np.zeros((0, self.n_dim + 1))
        self.storage['FPs'] = np.vstack([x for x in fps_cache]) \
            if len(fps_cache) > 0 else np.zeros((0, self.n_dim + 1))
        self.storage['IPs'] = np.vstack([x for x in ips_cache]) \
            if len(ips_cache) > 0 else np.zeros((0, self.n_dim + 1))

    def crossover(self):
        aps = self.storage['APs']
        aps_x = aps[:, :-1]
        aps_y = aps[:, -1]
        aps_y_scale = aps_y - aps_y.min() + 1e-8
        cr_prob = aps_y_scale / np.sum(aps_y_scale)
        n_cr = int(aps.shape[0] * self.pcr)
        # 偶数交叉+1比-1评价次数增多但可能效果更好
        n_cr = n_cr if n_cr % 2 == 0 else n_cr - 1
        cr_idx = np.random.choice(aps.shape[0], n_cr, p=cr_prob)
        cr_aps = aps_x[cr_idx]
        new_aps = np.zeros((n_cr // 2, self.n_dim + 1))
        for i in range(0, len(cr_aps), 2):
            new_aps[i // 2, :-1] = (self.theta * cr_aps[i] +
                                    (1 - self.theta) * cr_aps[i + 1])

        # 去除交叉产生的重复点
        # new_aps = np.unique(new_aps, axis=0)
        # new_aps = pd.DataFrame(new_aps).drop_duplicates().values.copy()

        # 评价适应度
        new_aps[:, -1] = self.fitness(new_aps[:, :-1])
        # 交叉后的点并入active points
        self.storage['APs'] = np.vstack([self.storage['APs'], new_aps])

    def selection(self):
        # 筛选较好的active points
        aps = self.storage['APs']
        aps_y = aps[:, -1]
        sel_num = int(aps.shape[0] * self.pa)
        sel_idx = np.argsort(aps_y)[::-1][:sel_num]
        non_sel_idx = np.argsort(aps_y)[::-1][:sel_num]
        self.storage['APs'] = aps[sel_idx]
        self.storage['IPs'] = np.vstack([self.storage['IPs'], aps[non_sel_idx]])

    def record_snap(self):
        snap_samples = self.storage.copy()
        self.snap.append(snap_samples)

    def run(self):
        # 初始化采样
        self.init_sampling()

        # 开始迭代
        for t in range(self.n_iter):
            if self.storage['APs'].shape[0] > 0:
                # 采样操作
                self.update_samples()
                # 交叉操作
                self.crossover()
                # 记录快照
                self.record_snap()
                # 选择操作
                self.selection()
                # 评价次数
                self.eval_count_list.append(self.eval_count)
            else:
                break

            if self.storage['APs'].shape[0] > 0:
                print(t, self.snap[-1]['APs'][:, -1].max())

        print('Num of Evaluation', self.eval_count)
        print('Max of GASO APs', np.max([x['APs'][:, -1].max() for x in self.snap
                                         if x['APs'].size > 0]))
        print('Max of GASO FPs', np.max([x['FPs'][:, -1].max() for x in self.snap
                                         if x['FPs'].size > 0]))

        _best_idx = [x['APs'][:, -1].argmax() for x in self.snap
                     if x['APs'].size > 0]
        history_best = [x['APs'][_best_idx[i]] for i, x in enumerate(self.snap)
                        if x['APs'].size > 0]
        self.history_best = np.vstack([x for x in history_best])
        best_idx = np.argmax(self.history_best[:, -1])
        x_best = self.history_best[best_idx, :-1]
        y_best = self.history_best[best_idx, -1]
        output = (x_best, y_best)
        self.eval_count_list = np.array(self.eval_count_list)
        return output


if __name__ == '__main__':
    # 可复现性
    SEED = 1024
    np.random.seed(SEED)

    # 超参数
    func = schaffer
    n_dim = 2
    lb = [-100, -100]
    ub = [100, 100]
    # lb = [0, 0]
    # ub = [10, 10]

    gaso_opt = GASO(func=func, n_dim=n_dim, n_iter=20, n_inits=80,
                    lb=lb, ub=ub, pcr=0.5, pse=0.8, psp=0.2,
                    n_samples=8, p_epsilon=0.05, theta=0.5)
    xmax, fval = gaso_opt.run()

    plt.figure()
    plt.plot(gaso_opt.history_best[:, -1], 'r',
             label='Max of GASO APs Generation')
    plt.legend()
    plt.show()

    APs = [x['APs'][:, :-1] for x in gaso_opt.snap]
    FPs = [x['FPs'][:, :-1] for x in gaso_opt.snap]
    IPs = [x['IPs'][:, :-1] for x in gaso_opt.snap]

    # a = AnimatedScatter(
    #     func=func,
    #     lb=lb,
    #     ub=ub,
    #     points_list=APs,
    #     n_grid=1000,
    #     # file_name='gaso_schaffer.gif',
    # )
    # plt.show()

    ode_loss_func(xmax, fig=True)
