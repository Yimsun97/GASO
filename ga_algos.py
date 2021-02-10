# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import spotpy as sp
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse


class GAOPT:
    """This is a class for Genetic Algorithm.

    Class GAOPT uses binary encoding, 2-point crossover and swap mutation.
    Selection methods include "Roulette" and "Tournament".

    Attributes:
        func:           target function of maximum
        n_dim:          dimension of independent variables
        n_iter:         number of iterations
        n_pop:          number of population
        pcr:            rate of crossover
        pm:             rate of mutation
        lb:             lower limits of independent variables
        ub:             upper limits of independent variables
        precision:      maximum precision of real number due to encoding
        sel:            selection method, including "Roulette" or "Tournament"
        interval_max:   maximum of intervals of independent variables
        int_bit:        bits of int
        _chrom_len:     length of chromosome
        history_best:   best individual of historic generations
        history_mean:   mean of fitness of historic generations

    """

    def __init__(self, func, n_dim, n_iter, n_pop,
                 pcr, pm, lb, ub, precision, sel='Tournament'):
        self.func = func
        self.n_dim = n_dim
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.pcr = pcr
        self.pm = pm
        self.lb = lb
        self.ub = ub
        self.precision = precision
        self.sel = sel

        self.interval_max = np.max(np.array(self.ub) -
                                   np.array(self.lb))
        self.int_bit = np.ceil(np.log2(self.interval_max))
        self._chrom_len = self.chrom_len()
        self.history_best = []
        self.history_mean = []

    def chrom_len(self):
        """Calculate the length needed according to precision.
        :return: length of chromosome
        """
        _chrom_len = np.ceil(
            np.log2(self.interval_max / self.precision + 1)
        )
        return np.int8(_chrom_len)

    def encoding(self):
        pass

    def decoding(self, encoding_chrom):
        """Decoding of chromosome.

        :param encoding_chrom: chromosomes before decoding
        :return: decoded chromosome
        """
        exponent = np.arange(self.int_bit - 1,
                             self.int_bit - self._chrom_len - 1,
                             step=-1)
        scaler = 2.0 ** exponent

        decoding_chrom = []
        for i, _chrom in enumerate(encoding_chrom):
            positive_chrom = np.sum(scaler * _chrom, axis=1)
            trans_chrom = (
                    self.lb[i] +
                    (self.ub[i] - self.lb[i]) / 2.0 ** self.int_bit
                    * positive_chrom
            )
            decoding_chrom.append(trans_chrom)
        return decoding_chrom

    def crossover(self, chrom):
        """Crossover operator.

        :param chrom: chromosomes before crossover
        :return: chromosomes after crossover
        """
        assert len(chrom[0].shape) >= 2, "input chromosome is binary encoded"

        # calculate number of chromosomes of crossover
        n_cr_chrom = int(self.n_pop * self.pcr)
        # ensure number of chromosomes of crossover is even
        if n_cr_chrom % 2 != 0:
            n_cr_chrom += 1

        # # randomly choose chromosomes of crossover
        # cr_idx = np.random.choice(self.n_pop, n_cr_chrom,
        #                           replace=False).reshape(-1, 2)
        # perform crossover according to order
        cr_idx = np.arange(n_cr_chrom).reshape(-1, 2)
        original_idx = [x for x in np.arange(self.n_pop)
                        if x not in cr_idx.ravel()]
        # 染色体交叉
        cr_chrom = []
        for _chrom in chrom:
            _cr_chrom = np.zeros((self.n_pop, self._chrom_len))
            for i in range(n_cr_chrom // 2):
                chrom_1 = _chrom[cr_idx[i, 0]]
                chrom_2 = _chrom[cr_idx[i, 1]]
                # select two crossover points
                cr_point = np.sort(
                    np.random.choice(self._chrom_len - 1,
                                     2, replace=False)
                )
                cr_slice = slice(cr_point[0], cr_point[1])
                chrom_1[cr_slice] = chrom_2[cr_slice].copy()
                chrom_2[cr_slice] = chrom_1[cr_slice].copy()
                _cr_chrom[2 * i] = chrom_1
                _cr_chrom[2 * i + 1] = chrom_2
                _cr_chrom[n_cr_chrom:] = _chrom[original_idx]
            cr_chrom.append(_cr_chrom)
        return cr_chrom

    def mutation(self, chrom):
        """Mutation operator.

        :param chrom: chromosomes before mutation
        :return: chromosomes after mutation
        """
        assert len(chrom[0].shape) >= 2, "input chromosome is binary encoded"

        def swap(single_chrom):
            mut_prob = np.random.rand(self.n_pop, )
            mut_idx = mut_prob < self.pm
            for i, true_idx in enumerate(mut_idx):
                if true_idx:
                    bit1, bit2 = np.random.choice(self._chrom_len, 2,
                                                  replace=False)
                    single_chrom[i, bit1] = single_chrom[i, bit2].copy()
                    single_chrom[i, bit2] = single_chrom[i, bit1].copy()
            return single_chrom

        mut_chrom = []
        for _chrom in chrom:
            _chrom = swap(_chrom)
            mut_chrom.append(_chrom)
        return mut_chrom

    def selection(self, chrom, ):
        """Select chromosomes using Roulette or Tournament

        :param chrom: chromosomes before selection
        :return: selected chromosomes
        """
        assert len(chrom[0].shape) >= 2, "input chromosome is binary encoded"
        # evaluate the fitness
        _fitness = self.fitness(chrom)
        # define selection method
        method = self.sel

        if method == 'Roulette':
            # Roulette selection
            # scale the fitness
            _fitness = _fitness - _fitness.min() + 1e-8
            sum_fitness = np.sum(_fitness)
            select_prob = _fitness / sum_fitness
            # restore the elite individuals
            elite_idx = _fitness.argsort()[::-1][:int(self.n_pop * 0.04)]
            roulutte_idx = np.random.choice(self.n_pop,
                                            self.n_pop - len(elite_idx),
                                            p=select_prob)
            select_idx = np.concatenate([elite_idx, roulutte_idx])
            select_chrom = [x[select_idx.astype(int)]
                            for x in chrom]
        elif method == 'Tournament':
            # Tournament selection
            tourn_size = int(self.n_pop * 0.1)
            select_idx = []
            for i in range(self.n_pop):
                # aspirants_index = np.random.choice(range(self.n_pop), size=tourn_size)
                aspirants_idx = np.random.randint(self.n_pop, size=tourn_size)
                select_idx.append(max(aspirants_idx, key=lambda x: _fitness[x]))
            select_chrom = [x[select_idx] for x in chrom]
        else:
            raise ValueError("No such method")

        return select_chrom

    def fitness(self, chrom):
        """Evaluate the fitness of every individual.

        :param chrom: chromosomes to evaluate
        :return: fitness of chromosomes without scale
        """
        assert len(chrom[0].shape) >= 2, "Chromosome should be decoded before evaluation"
        _chrom = self.decoding(chrom)
        _fitness = self.func([*_chrom]).ravel()

        def record():
            best_idx = np.argmax(_fitness)
            x_best = [x[best_idx] for x in _chrom]
            y_best = _fitness[best_idx]
            self.history_best.append((*x_best, y_best))
            self.history_mean.append(np.mean(_fitness))

        record()
        return _fitness

    def init_pop(self):
        """Initialize the population randomly.

        :return: initial population
        """
        _chrom_len = self.chrom_len()

        init_chrom = []
        for i in range(self.n_dim):
            init_chrom.append(
                np.random.randint(2, size=(self.n_pop,
                                           _chrom_len))
            )
        return init_chrom

    def run(self):
        """Start iteration.

        :return: output of the optimization, sko_best_x refers to max value points
        sko_best_y refers to max value
        """
        # initialization
        iter_chrom = self.init_pop()

        # begin iterations
        for t in range(self.n_iter):
            # selection
            iter_chrom = self.selection(iter_chrom)
            # crossover
            iter_chrom = self.crossover(iter_chrom)
            # mutation
            iter_chrom = self.mutation(iter_chrom)

        # output history records
        self.history_best = np.array(self.history_best)
        best_idx = np.argmax(self.history_best[:, -1])
        x_best = self.history_best[best_idx, :-1]
        y_best = self.history_best[best_idx, -1]
        output = (x_best, y_best)
        return output


# 导入自定义问题接口
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, func, Dim, lb, ub, n_group):
        # 存储func
        self.func = func
        self.Dim = Dim
        self.n_group = n_group

        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # Dim = 1 # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        # lb = [-1]  # 决策变量下界
        # ub = [2]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        x = np.zeros((self.n_group, 1))

        i = 0
        for value in Vars:
            x[i] = self.func(value)
            i += 1

        pop.ObjV = x  # 计算目标函数值，赋值给pop种群对象的ObjV属性


def geatpy_optimizer(func, Dim, lb, ub, n_group, n_iter, pco, pm, seed):
    """ 定义优化问题

    """

    # 随机数种子设置
    np.random.seed(seed=seed)

    """===============================实例化问题对象==========================="""
    problem = MyProblem(func, Dim, lb, ub, n_group)  # 生成问题对象
    """=================================种群设置=============================="""
    Encoding = 'BG'  # 编码方式
    NIND = n_group  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = n_iter  # 最大进化代数
    myAlgorithm.recOper.XOVR = pco  # 设置交叉概率
    myAlgorithm.mutOper.Pm = pm  # 设置变异概率
    myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    # 保留前5个最优的解
    # top_five_index = np.argsort(population.ObjV * problem.maxormins, axis=0)[:5]
    # top_five_pop = population[top_five_index]
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """=================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
        print('最优的控制变量值为：')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
    else:
        print('没找到可行解。')

    return myAlgorithm, population


class spot_setup(object):
    """
    A 3 dimensional implementation of the Rosenbrock function
    Result at (1,1,1) is 0.
    """
    x1 = Uniform(0, 2, 1.5, 3.0, -100, 100, doc='x1 value of Rosenbrock function')
    x2 = Uniform(0, 2, 1.5, 3.0, -100, 100, doc='x2 value of Rosenbrock function')
    x3 = Uniform(0, 2, 1.5, 3.0, -100, 100, doc='x3 value of Rosenbrock function')
    x4 = Uniform(0, 2, 1.5, 3.0, -100, 100, doc='x4 value of Rosenbrock function')
    x5 = Uniform(0, 2, 1.5, 3.0, -100, 100, doc='x5 value of Rosenbrock function')
    x6 = Uniform(0, 2, 1.5, 3.0, -100, 100, doc='x6 value of Rosenbrock function')

    def __init__(self, obj_func=None):
        self.obj_func = obj_func

    def simulation(self, vector):
        x = np.array(vector)
        simulations = [sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)]
        return simulations

    def evaluation(self):
        observations = [0]
        return observations

    def objectivefunction(self, simulation, evaluation, params=None):

        # SPOTPY expects to get one or multiple values back,
        # that define the performence of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure on flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(1024)

    f = lambda x: x[0] * np.sin(10 * np.pi * x[1]) + 2.0
    results, _ = geatpy_optimizer(f, 2, [-1, -1], [2, 2], 40, 25, 0.7, 0.01, 1024)

    sampler = sp.algorithms.sceua(spot_setup())             # Initialize your model with a setup file
    sampler.sample(5000, ngs=7)                             # Run the model
    res_sceua = sampler.getdata()                           # Load the results
    sp.analyser.plot_parametertrace(res_sceua)              # Show the results
    sol = sp.analyser.get_best_parameterset(res_sceua, maximize=False)
    sol = np.array(list(sol[0]))
