# coding: utf8
import numpy as np
from cvxopt import matrix, solvers
import scipy as sp


def min_volatility_weighted(df):
    """
    波动率最小化加权 (w1 * df.col1 + w2 * df.col2 + ...) ** 2 ==> sum ==> min
    Args:
        df(DataFrame): 一只或多只基金收益, 行为某期(日度/周度/月度等), 列为基金
    Returns:
        list: 权重
    Examples:
        df = pd.DataFrame(np.random.randn(10,3), columns=['fund1', 'fund2', 'fund3'])
        min_volatility_weighted(df)
    Notes:
        二次规划标准形式为
        min 1/2 x^{T} P x + q^{T} x
        s.t. G x <= h
             A x = b

        本函数的形式为:
        min sum((df * w - (df * w).mean()) ** 2) # 方差
        s.t. \sum w_{i} = 1
        s.t. w_{i} >= 0
    """

    m, n = df.shape
    # M=df.values
    # Mx - \bar{Mx}
    mat = (np.eye(m) - 1.0 / m * np.ones((m * m, 1)).reshape(m, m)).dot(df.values)

    P = matrix(mat.T.dot(mat))
    A = matrix(np.ones((1, n)))
    b = matrix([[1.0]])
    G = matrix(-np.eye(n))
    q = matrix(np.zeros((n, 1)))
    h = matrix(np.zeros((n, 1)))
    sol = solvers.qp(P, q, G, h, A, b)
    return list(sol['x'])


def max_non_relative_weighted(df):
    """
    最大各列不相关加权. max \sum w1 * w2 * corr(col1, col2) + w2 * w3 * corr(col2, col3) + ... + wn * w1 corr(coln, col1)
    Args:
        df(DataFrame): 基金收益率, 行为日期, 列为基金

    Returns:
        list: 权重
    Notes:

    """
    M = df.corr().values
    m, n = M.shape
    mat = np.zeros_like(M)
    rows = np.array([i for i in range(0, m - 1)] + [0])
    cols = np.array([i for i in range(1, m)] + [-1])
    mat[rows, cols] = M[rows, cols]
    mat[cols, rows] = M[cols, rows]

    P = matrix(-mat)
    A = matrix(np.ones((1, n)))
    b = matrix([[1.0]])
    G = matrix(-np.eye(n))
    q = matrix(np.zeros((n, 1)))
    h = matrix(np.zeros((n, 1)))
    sol = solvers.qp(P, q, G, h, A, b)
    return list(sol['x'])


def max_sharpe_weighted(df):
    """
    最大化 收益/风险
    Args:
        df(DataFrame): 基金收益率, 行为日期, 列为基金
    Returns:
        list: 权重
    """
    M = df.values
    nrow, ncol = M.shape

    prime = np.eye(ncol)

    def generate_initial_value(ncol):
        """
        产生一随机初始值.这一随机初始值和为0, wi>=0.
        Args:
            ncol(int), 列数,等于未知数的个数
        Returns:
            np.array: 一组随机初始值
        """
        pre = list(np.random.uniform(0.0, 1.0 / ncol, size=ncol - 1))
        last = 1 - sum(pre)
        initial_value = np.array(pre + [last])
        return initial_value

    def variance_over_mean(x):
        """
        本应计算max np.mean(Mx)/np.std(Mx) over x, 但是优化函数的标准形式是最小化, 故函数写成其倒数，又为了求导方便, 使用
         variance 而不是std.
        Args:
            x(array-like): 权重

        Returns:
            float: 夏普的相反数
        """
        w = np.array(x)
        s = M.dot(w)
        return np.var(s) / np.mean(s)

    def jac(x):
        """
        计算上面函数sharpe的导数
        Args:
            x (array-like): 权重
        Returns:
            np.array, 导数
        Notes:
            let y = M \dot x
            then sharpe value F equals:
            $$ \frac{\sum y_{i}^{2}}{{\bar{y}}} - \bar{y} $$

            so F partial x is
            $$ \frac{\partial F}{\partial x_i} = \frac{\partial F}{\partial y_1} \frac{\partial y_1}{\partial x_i} +
             \frac{\partial F}{\partial y_2} \frac{\partial y_2}{\partial x_i} +
              \cdots +
              \frac{\partial F}{\partial y_n} \frac{\partial y_n}{\partial x_i}  $$

            and,
            $$
            \frac{\partial F} {\partial y_k} = \frac{ 2y_k \bar{y} - \frac{\sum y_i^2}{n}} {\bar{y}^2}  - \frac{1}{n}
            $$.

            $$
            \frac{\partial y_k}{\partial x_i} = M_{ki}
            $$
        """
        y = M.dot(x)
        ybar = np.mean(y)
        sum_y2 = np.sum(y ** 2)
        F2y = [(2 * y[k] * ybar - sum_y2 / nrow) / ybar ** 2 - 1.0 / nrow for k in range(nrow)]
        F2y = np.array(F2y)
        return F2y.dot(M)

    con_eq = [{'type': 'eq',
               'fun': lambda x: 1.0 * np.array(np.sum(x) - 1),
               'jac': lambda x: 1.0 * np.ones_like(x)}]

    con_ineq = [{'type': 'ineq',
                 'fun': lambda x: 1.0 * np.array(x[i]),
                 'jac': lambda x: 1.0 * prime[i]}
                for i in range(ncol)
                ]
    cons = tuple(con_eq + con_ineq)
    bounds = [(0, 1)] * ncol
    initial_value = generate_initial_value(ncol=ncol)
    count = 1000
    while count > 0:
        count -= 1
        opt = sp.optimize.minimize(variance_over_mean, initial_value, jac=jac, method='SLSQP', constraints=cons, bounds=bounds,
                                   options={'disp': True, 'maxiter': 10000})
        if np.isclose(np.sum(opt.x), 1) and np.all(opt.x > 0) and opt.success:
            break
        else:
            initial_value = generate_initial_value(ncol=ncol)
    if count <= 0:
        raise ValueError('maximum tries 1000 exceed. please try again!')
    return opt.x.tolist()
