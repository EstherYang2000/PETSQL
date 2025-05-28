import logging
import random

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import math

def auto_select_epsilon(num_experts, num_rounds=None, best_mistakes=None):
    """
    自動計算 epsilon：
    - 若提供 best_mistakes（M*），使用理論最優值
    - 否則退回保守估計 ε = sqrt(log(N)/T)
    """
    if num_experts < 1:
        raise ValueError("num_experts 必須 >= 1")
    
    log_N = math.log(num_experts)

    if best_mistakes is not None and best_mistakes > 0:
        return math.sqrt(log_N / best_mistakes)
    elif num_rounds is not None and num_rounds > 0:
        return math.sqrt(log_N / num_rounds)
    else:
        raise ValueError("需提供 num_rounds 或 best_mistakes 其中之一作為依據")


class WeightedMajorityAlgorithm:
    """
    實作 WMA (Weighted Majority Algorithm) 的簡易版本：
      - 每位「專家」(expert) 都有一個權重
      - 若該專家產生的最終預測是錯誤的，就降低 (衰減) 其權重
      - 若正確，則保持或輕微增幅 (本範例預設不增，只維持)
    """

    def __init__(self, experts=None, epsilon=0.005):
        """
        初始化 WMA 演算法。
        
        :param experts: dict 或 None
            - 若為 dict, 形如 {"expert_name1": 1.0, "expert_name2": 1.0, ...}
            - 若為 None, 則預設為空，在運行時再動態加入專家
        :param epsilon: float, 衰減比例(0<epsilon<1)，專家預測錯誤時要乘的因子 (1 - epsilon)
        """
        if experts is None:
            experts = {}
        self.experts = experts      # {expert_name: weight}
        self.epsilon = epsilon      # 衰減比例
        self.mistake_counter = {name: 0 for name in experts}

    def add_expert(self, expert_name: str, init_weight: float = 1.0, init_mistake_counter: int = 0):
        """
        Add a new expert to the algorithm with an initial weight.
        
        Args:
            expert_name (str): Name/identifier of the expert
            init_weight (float): Initial weight for the expert (default: 1.0)
            init_mistake_counter (int): Initial mistake count for the expert (default: 0)
        """
        if expert_name not in self.experts:
            self.experts[expert_name] = init_weight
            self.mistake_counter[expert_name] = init_mistake_counter

    def reset_mistake_counts(self):
        """重置所有專家的錯誤計數"""
        self.mistake_counter = {name: 0 for name in self.experts}

    def update_mistake_count(self, expert_name: str):
        """只更新錯誤計數，不更新權重"""
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in the algorithm")
        self.mistake_counter[expert_name] += 1

    def update_weights(self, expert_name: str, is_correct: bool, strategy="wma"):
        """
        Update the weight of a single expert based on their prediction correctness.
        
        Args:
            expert_name (str): The name of the expert whose weight should be updated
            is_correct (bool): Whether the expert's prediction was correct
            strategy (str): The strategy to use for updating weights ("wma", "rwma", "naive", "rl")
        """
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in the algorithm")

        if strategy == "wma" or strategy == "rwma":
            if not is_correct:
                # 更新權重
                old_weight = self.experts[expert_name]
                self.experts[expert_name] = old_weight * (1 - self.epsilon)
                # 更新錯誤計數
                self.mistake_counter[expert_name] += 1
        elif strategy == "naive":
            # 只記錄錯誤，不更新權重
            if not is_correct:
                self.mistake_counter[expert_name] += 1
        # 對於 "rl" 策略，不更新權重也不更新錯誤計數

    def get_mistake_counts(self):
        """
        回傳所有專家的錯誤次數。
        """
        return self.mistake_counter.copy()

    def weighted_majority_vote(self, predictions_dict):
        """
        Perform a weighted majority vote where each expert provides a single SQL.
        
        :param predictions_dict: dict
            { 
                "expert_1": ["SQL A", "SQL B"],
                "expert_2": ["SQL B", "SQL C"],
                ...
            }
        :return: (best_sql, chosen_experts, best_weight)
            - best_sql: 獲勝SQL
            - chosen_experts: 推薦這條SQL的專家列表
            - best_weight: 這條SQL累積的總加權分數
        """
        sql_to_weight = {}
        sql_to_experts = {}

        # Check if predictions_dict is empty
        if not predictions_dict:
            logger.error("No SQL predictions received from any expert.")
            return None, [], 0.0
        
        # Accumulate weights for each SQL
        for expert_name, sql_str in predictions_dict.items():
            expert_weight = self.experts.get(expert_name, 1.0)
            # 每條候選SQL都獲得該專家的全部權重 (Group Voting)
            if sql_str not in sql_to_weight:
                sql_to_weight[sql_str] = 0.0
                sql_to_experts[sql_str] = []
            sql_to_weight[sql_str] += expert_weight
            sql_to_experts[sql_str].append(expert_name)
        # If no valid SQLs were added, return a safe fallback
        if not sql_to_weight:
            logger.error("No valid SQLs received from experts.")
            return None, [], 0.0
        # 選出加權分數最高的SQL
        best_sql = max(sql_to_weight, key=sql_to_weight.get)
        best_weight = sql_to_weight[best_sql]
        chosen_experts = list(set(sql_to_experts[best_sql]))

        return best_sql, chosen_experts, best_weight

    def randomized_weighted_majority_vote(self, predictions_dict):
        """
        Randomized version: draw a SQL according to expert weights.
        """
        if not predictions_dict:
            logger.error("No SQL predictions received from any expert.")
            return None, [], 0.0

        # 1. Normalize weights into probabilities
        total_weight = sum(self.experts.get(expert, 0.0) for expert in predictions_dict)
        if total_weight == 0:
            logger.warning("Total expert weight is zero. Falling back to uniform.")
            probs = {expert: 1.0 / len(predictions_dict) for expert in predictions_dict}
        else:
            probs = {expert: self.experts.get(expert, 0.0) / total_weight for expert in predictions_dict}

        # 2. Randomly select an expert by probability
        selected_expert = random.choices(
            population=list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]

        # 3. From that expert, randomly select one SQL (or top-1)
        final_sql = predictions_dict[selected_expert]
        if not final_sql:
            logger.warning(f"Selected expert {selected_expert} has no predictions.")
            return None, [], 0.0
        # final_sql = random.choice(sql_list)

        return final_sql, [selected_expert], self.experts[selected_expert]
    
    def get_weights(self):
        """
        Get current weights of all experts.
        
        Returns:
            dict: Dictionary mapping expert names to their current weights
        """
        return self.experts.copy()

    def get_expert_weight(self, expert_name: str) -> float:
        """
        Get the current weight of a specific expert.
        
        Args:
            expert_name (str): Name of the expert
            
        Returns:
            float: Current weight of the expert
            
        Raises:
            ValueError: If expert not found
        """
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in the algorithm")
        return self.experts[expert_name]