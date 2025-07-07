from .base import BaseLearner
from typing import Literal
import numpy as np
import streamlit as st
import pylibbandit.environments

class EXP3(BaseLearner):

    def __init__(self, eta: float|Literal['auto'] = 'auto'):
        super().__init__()
        self.auto_eta = eta == 'auto'
        if not self.auto_eta:
            self.eta = eta

    def decide(self, n: int, t: int, k: int, batch_size: int) -> np.ndarray[int]:
        if t == 1:
            self.k = k
            self.batch_size = batch_size
            self.S = np.zeros((batch_size, k), dtype=float)
            if self.auto_eta:
                self.eta = np.sqrt(2 * np.log(k) / (k * n))

        logits = self.eta * self.S
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return np.array([np.random.choice(k, p=self.probs[i]) for i in range(batch_size)])

    def reward(self, x: np.ndarray, t: int, i: np.ndarray):
        # Normalize rewards to [0, 1]
        x = np.clip(x, 0, 1)
        for b in range(self.batch_size):
            for a in range(self.k):
                if a == i[b]:
                    self.S[b, a] += 1 - ((1 - x[b]) / self.probs[b, a])
                else:
                    self.S[b, a] += 1

    @classmethod
    def display_name(cls):
        return "EXP3 (log-space)"

    @classmethod
    def st_config(cls, prefix):
        return {
            "eta": st.number_input("Learning rate η", min_value=0.0, value=None, placeholder="Leave empty for automatic η", key=prefix+"eta")
        }

    @classmethod
    def validate_update_config(cls, config: dict, instance: "pylibbandit.environments.StochasticEnvironment"):
        if not config["eta"]: del config["eta"]
        return config
