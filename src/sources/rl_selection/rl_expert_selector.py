# expert_env.py
import torch
import numpy as np
from collections import deque
from typing import List, Callable
import os

class ExpertSelectionEnv:
    """
    強化學習環境：每步選擇一位專家，回傳 embedding 作為 observation，
    reward 由外部 batched_evaluate_fn 計算。
    """
    def __init__(
        self,
        prompts: List[str],
        experts: List[dict],
        gold_sqls: List[str],
        prompt_embs: torch.Tensor,          # 預先算好的 (N, emb_dim)
        batched_evaluate_fn: Callable,
        kmaps,
        db_path: str,
        dataset_type: str,
        diversity_bonus: float = 0.1,
        diversity_window: int = 10,
    ):
        self.prompts          = prompts
        self.experts          = experts
        self.gold_sqls        = gold_sqls
        self.prompt_embs      = prompt_embs  # 已在 GPU/CPU
        self.evaluate_fn      = batched_evaluate_fn
        self.kmaps            = kmaps
        self.db_path          = db_path
        self.dataset_type     = dataset_type

        self.bonus            = diversity_bonus
        self.recent_actions   = deque(maxlen=diversity_window)
        self.n_experts        = len(experts)
        self.idx              = 0

    # -------- Gym-like API --------
    def reset(self):
        self.idx = 0
        self.recent_actions.clear()
        return self.prompt_embs[self.idx]

    def step(self, action: int):
        pred_sql  = self.experts[action]["raw_sql_outputs"][self.idx]
        gold_sql  = self.gold_sqls[self.idx]
        # 使用 partial match reward
        from evaluation import evaluate_partial,evaluate_cc
        partial_scores = evaluate_partial(gold_sql, pred_sql, self.db_path, self.kmaps)
        # 權重組合
        def get_f1(partial_scores, key):
            return partial_scores.get(key, {}).get('f1', 0)
        partial_f1 = (
            0.14 * get_f1(partial_scores, 'select') +
            0.10 * get_f1(partial_scores, 'select(no AGG)') +
            0.14 * get_f1(partial_scores, 'where') +
            0.10 * get_f1(partial_scores, 'where(no OP)') +
            0.08 * get_f1(partial_scores, 'group(no Having)') +
            0.06 * get_f1(partial_scores, 'group') +
            0.10 * get_f1(partial_scores, 'order') +
            0.08 * get_f1(partial_scores, 'and/or') +
            0.10 * get_f1(partial_scores, 'IUEN') +
            0.10 * get_f1(partial_scores, 'keywords')
        )
        # 分層 reward 設計
        em = evaluate_cc(gold_sql, pred_sql, self.db_path, self.dataset_type, self.kmaps)
        # exec match 判斷（假設 evaluate_cc 只回傳 1/0，若需 exec match，請用 exec_match function 補充）
        if em == 1:
            reward = 1.2
        else:
            reward = 0.3 * partial_f1

        # diversity bonus
        diversity = self.bonus if action not in self.recent_actions else 0.0
        reward += diversity
        # penalty for repeated wrong choices
        if action in self.recent_actions and reward < 1.0:
            reward -= 0.1  # penalty value，可依需求調整
        self.recent_actions.append(action)

        self.idx += 1
        done = self.idx >= len(self.prompts)
        obs  = None if done else self.prompt_embs[self.idx]
        return obs, reward, done, {}

    # --------- utility for inference ---------
    def get_obs(self, idx):
        return self.prompt_embs[idx]
