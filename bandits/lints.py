import numpy as np

class LinTSAlgorithm:
    """
    Linear Thompson Sampling 算法的核心实现。
    该实现改编自 modelselection-main/bandit_algs.py 中的 LUCBalgorithm，
    修改了其臂选择逻辑，从构造置信上界 (UCB) 变为纯粹的后验采样 (TS)。
    """
    def __init__(self, dimension: int, alpha: float = 1.0, lambda_reg: float = 1.0):
        """
        初始化 LinTS 算法。

        Parameters
        ----------
        dimension : int
            上下文特征的维度。
        alpha : float, optional
            探索参数，用于缩放采样时的协方差。默认为 1.0。
        lambda_reg : float, optional
            正则化参数，用于初始化协方差矩阵。默认为 1.0。
        """
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # A: (d x d) 矩阵，等于 B.T * B + lambda * I
        self.A = np.eye(dimension) * self.lambda_reg
        # b: (d x 1) 向量，等于 B.T * y
        self.b = np.zeros(dimension)
        
        # theta_hat 是对参数向量的估计
        self.theta_hat = np.zeros(dimension)

    def reset(self):
        """Resets the algorithm's statistics to their initial state."""
        self.A = np.eye(self.dimension) * self.lambda_reg
        self.b = np.zeros(self.dimension)
        self.theta_hat = np.zeros(self.dimension)

    def update(self, context: np.ndarray, reward: float):
        """
        使用观察到的奖励更新算法参数。

        Parameters
        ----------
        context : np.ndarray
            上下文特征向量 (d, )。
        reward : float
            观察到的奖励。
        """
        self.A += np.outer(context, context)
        self.b += reward * context
        
        # 通过求解 A * theta = b 来更新 theta_hat
        A_inv = np.linalg.inv(self.A)
        self.theta_hat = A_inv @ self.b

    def sample_and_get_rewards(self, contexts: np.ndarray) -> np.ndarray:
        """
        从后验分布中采样一个参数向量，并用它来计算给定的一组上下文的期望奖励。

        Parameters
        ----------
        contexts : np.ndarray
            一个 (n_arms, d) 的矩阵，包含所有待评估臂的上下文。

        Returns
        -------
        np.ndarray
            一个 (n_arms, ) 的向量，包含每个臂的估计奖励。
        """
        try:
            A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(self.A) # 使用伪逆以保证数值稳定性

        # 从后验分布 N(theta_hat, alpha^2 * A_inv) 中采样一个 theta_tilde
        theta_tilde = np.random.multivariate_normal(
            self.theta_hat, 
            self.alpha**2 * A_inv
        )
        
        # 计算每个臂的期望奖励
        estimated_rewards = contexts @ theta_tilde
        return estimated_rewards


class LinTS:
    """
    一个适配器类，用于包装 LinTSAlgorithm，使其符合项目统一的接口。
    """
    def __init__(self, d: int, nu: float = 1.0, lambda_reg: float = 1.0, **kwargs):
        """
        初始化 LinTS 适配器。

        Parameters
        ----------
        d : int
            上下文特征的维度。
        nu : float, optional
            探索参数 (ν)，用于缩放采样时的协方差。默认为 1.0。
        lambda_reg : float, optional
            正则化参数 (λ)。默认为 1.0。
        """
        self.d = d
        self._algorithm = LinTSAlgorithm(dimension=d, alpha=nu, lambda_reg=lambda_reg)

    def reset(self):
        """Resets the underlying LinTS algorithm."""
        self._algorithm.reset()

    def select_arm(self, context: np.ndarray) -> int:
        """
        根据 Thompson Sampling 原则选择一个臂。

        Parameters
        ----------
        context : np.ndarray
            上下文信息，形状为 (n_arms, n_dims)，其中 n_arms 是当前轮次可选的臂数量。
            每一行代表一个臂的上下文特征。

        Returns
        -------
        int
            被选中的臂的索引（在传入的 context 矩阵中的行索引）。
        """
        if context.ndim == 1:
             # 如果只有一个臂的上下文被提供
            context = context.reshape(1, -1)
        
        estimated_rewards = self._algorithm.sample_and_get_rewards(context)
        return np.argmax(estimated_rewards)

    def update(self, arm_context: np.ndarray, reward: float):
        """
        更新算法参数。

        Parameters
        ----------
        arm_context : np.ndarray
            被选中臂的上下文特征向量。
        reward : float
            观察到的奖励。
        """
        self._algorithm.update(context=arm_context, reward=reward) 