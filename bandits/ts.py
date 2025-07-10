import numpy as np
from .base_bandit import BaseBandit

class ThompsonSampling(BaseBandit):
    """
    Thompson Sampling algorithm for Bernoulli distributed rewards.
    """
    def __init__(self, n_arms: int, alpha: float = 1.0, beta: float = 1.0):
        """
        初始化 Thompson Sampling 算法。

        Parameters
        ----------
        n_arms : int
            臂的数量。
        alpha : float, optional
            Beta 分布的 alpha 先验参数, 默认为 1.0。
        beta : float, optional
            Beta 分布的 beta 先验参数, 默认为 1.0。
        """
        self.n_arms = n_arms
        self.initial_alpha = alpha
        self.initial_beta = beta
        # 为每个臂初始化 Beta 分布的先验参数
        self.alpha = np.full(n_arms, self.initial_alpha)
        self.beta = np.full(n_arms, self.initial_beta)

    def reset(self):
        """Resets the Beta distribution parameters for each arm to their initial prior state."""
        self.alpha = np.full(self.n_arms, self.initial_alpha)
        self.beta = np.full(self.n_arms, self.initial_beta)

    def select_arm(self, context=None) -> int:
        """
        根据 Thompson Sampling 原则选择一个臂。上下文被忽略。

        Parameters
        ----------
        context : Any, optional
            上下文信息，此处未使用。

        Returns
        -------
        int
            被选中的臂的索引。
        """
        # 从每个臂的 Beta 分布中采样一个概率
        sampled_probas = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        
        # 返回具有最高采样概率的臂的索引
        return np.argmax(sampled_probas)

    def update(self, arm: int, reward: float, context=None):
        """
        更新被选中的臂的 Beta 分布参数。

        Parameters
        ----------
        arm : int
            被选中的臂的索引。
        reward : float
            观察到的二元奖励 (0 或 1)。
        context : Any, optional
            上下文信息，此处未使用。
        """
        if reward == 1:
            # 如果奖励为 1，增加 alpha (成功次数)
            self.alpha[arm] += 1
        else:
            # 如果奖励为 0，增加 beta (失败次数)
            self.beta[arm] += 1
