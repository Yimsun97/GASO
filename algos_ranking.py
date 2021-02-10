import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from opt_utils import *
from vis_utils import vis_iter, vis_iter_agg
from GA_sampler import GASO
from sko.GA import GA
from ga_algos import GAOPT, geatpy_optimizer

# %% Global definition
SEED = 1024
n_dim = 10
func = lambda x: schaffer_nd(x, n=n_dim)
n_pop = 50
n_iter = 50
lb = [-10] * n_dim
ub = [10] * n_dim
# lb = [0] * n_dim
# ub = [10] * n_dim
# %% sko GA optimizer
np.random.seed(SEED)
skoga = GA(
    func=lambda x: -func(x)[0] if func == ode_loss_func else -func(x),
    n_dim=n_dim,
    size_pop=n_pop,
    max_iter=n_iter,
    prob_mut=0.05,
    lb=lb,
    ub=ub,
    precision=1e-5
)

sko_best_x, sko_best_y = skoga.run()
Y_history = pd.DataFrame(skoga.all_history_Y)

# %% Self defined GA optimizer
np.random.seed(SEED)
ga_opt = GAOPT(
    func=func,
    n_dim=n_dim,
    n_iter=n_iter,
    n_pop=n_pop,
    pcr=0.9,
    pm=0.01,
    lb=lb,
    ub=ub,
    precision=1e-5,
    sel="Tournament",
)
ga_opt_x, ga_opt_fval = ga_opt.run()
print(ga_opt_x, ga_opt_fval)

# %% Geatpy GA optimizer
geat_res, _ = geatpy_optimizer(
    func=func,
    Dim=n_dim,
    lb=lb,
    ub=ub,
    n_group=n_pop,
    n_iter=n_iter,
    pco=0.7,
    pm=0.05,
    seed=SEED
)
geat_best_x = geat_res.BestIndi.Phen.ravel()
geat_best_y = geat_res.BestIndi.ObjV.ravel()

# %% GASO GA optimizer
np.random.seed(SEED)
gaso_opt = GASO(
    func=func,
    n_dim=n_dim,
    n_iter=20,
    n_inits=80,
    lb=lb,
    ub=ub,
    pcr=0.7,
    pse=0.7,
    psp=0.3,
    # pcr=0.5,
    # pse=0.8,
    # psp=0.2,
    n_samples=8,
    p_epsilon=0.05,
    theta=0.5
)
gaso_best_x, gaso_best_y = gaso_opt.run()

# %% Summary
eval_counts = [
    n_pop * (np.arange(n_iter) + 1),
    n_pop * (np.arange(n_iter) + 1),
    n_pop * (np.arange(n_iter) + 1),
    gaso_opt.eval_count_list
]
iter_history = [
    Y_history.min(axis=1).cummin(),
    pd.DataFrame(-ga_opt.history_best[:, -1]).cummin().values,
    -np.array(geat_res.log['f_max']),
    -gaso_opt.history_best[:, -1],
]
target = 'Max'
legends = [
    f'{target} of sko',
    f'{target} of GAOPT',
    f'{target} of Geatpy',
    f'{target} of GASO',
]
vis_iter_agg(eval_counts, [-x for x in iter_history],
             legends, scale='linear')

# 分别可视化
# vis_iter(Y_history.min(axis=1).cummin(), 'Min of sko')
# vis_iter(ga_opt.history_best[:, -1], 'Max of GAOPT')
# vis_iter(geat_res.log['f_max'], "Max of Geatpy")
# vis_iter(gaso_opt.history_best[:, -1], 'Max of GASO')

if func == ode_loss_func:
    ode_loss_func(sko_best_x, fig=True)
    ode_loss_func(ga_opt_x, fig=True)
    ode_loss_func(geat_best_x, fig=True)
    ode_loss_func(gaso_best_x, fig=True)
