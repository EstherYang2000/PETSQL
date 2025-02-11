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

    def add_expert(self, expert_name: str, init_weight: float = 1.0):
        """
        Add a new expert to the algorithm with an initial weight.
        
        Args:
            expert_name (str): Name/identifier of the expert
            init_weight (float): Initial weight for the expert (default: 1.0)
        """
        if expert_name not in self.experts:
            self.experts[expert_name] = init_weight


    def update_weights(self, expert_name: str, is_correct: bool):
        """
        Update the weight of a single expert based on their prediction correctness.
        
        Args:
            expert_name (str): The name of the expert whose weight should be updated
            is_correct (bool): Whether the expert's prediction was correct
        """
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in the algorithm")

        if not is_correct:
            # Only decrease weight if the prediction was incorrect
            old_weight = self.experts[expert_name]
            self.experts[expert_name] = old_weight * (1 - self.epsilon)

    def weighted_majority_vote(self, predictions_dict):
        """
        Perform a weighted majority vote where each expert provides a single SQL.
        
        :param predictions_dict: dict
            { "expert_name1": "SQL1", "expert_name2": "SQL2", ... }
        :return: (best_sql, chosen_experts, best_weight)
            - best_sql: SQL with the highest total weight.
            - chosen_experts: List of experts who predicted the best_sql.
            - best_weight: Total weight of the best_sql.
        """
        sql_to_weight = {}
        sql_to_experts = {}

        # Accumulate weights for each SQL
        for expert_name, sql_str in predictions_dict.items():
            weight = self.experts.get(expert_name, 1.0)
            if sql_str not in sql_to_weight:
                sql_to_weight[sql_str] = 0.0
                sql_to_experts[sql_str] = []
            sql_to_weight[sql_str] += weight
            sql_to_experts[sql_str].append(expert_name)

        # Find the SQL with the highest weight
        best_sql = max(sql_to_weight, key=sql_to_weight.get)
        best_weight = sql_to_weight[best_sql]
        chosen_experts = sql_to_experts[best_sql]

        return best_sql, chosen_experts, best_weight

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
            ValueError: If expert_name is not found
        """
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in the algorithm")
        return self.experts[expert_name]