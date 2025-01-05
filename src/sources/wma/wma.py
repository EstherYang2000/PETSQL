class WeightedMajorityAlgorithm:
    """
    實作 WMA (Weighted Majority Algorithm) 的簡易版本：
      - 每位「專家」(expert) 都有一個權重
      - 若該專家產生的最終預測是錯誤的，就降低 (衰減) 其權重
      - 若正確，則保持或輕微增幅 (本範例預設不增，只維持)
    """

    def __init__(self, experts=None, epsilon=0.2):
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

    def add_expert(self, expert_name, init_weight=1.0):
        """
        新增一位專家，並賦予初始權重。
        """
        self.experts[expert_name] = init_weight

    def update_weights(self, chosen_experts, is_correct):
        """
        依照 WMA 的規則，更新被選上(即產生最終 SQL)之專家的權重。
        
        :param chosen_experts: list of str
            - 代表在本回合「最終預測」中被採用的專家名稱。
        :param is_correct: bool
            - 這個最終預測是否正確。
        """
        if is_correct:
            # 預設做法：若最終預測正確 -> chosen_experts 權重不變
            # 其他專家也不動。若你想懲罰那些沒參與到正解的專家，
            # 或者微增 chosen_experts，也可在此修改。
            pass
        else:
            # 若最終預測錯誤 -> chosen_experts 的權重衰減
            for expert_name in chosen_experts:
                old_w = self.experts.get(expert_name, 1.0)
                self.experts[expert_name] = old_w * (1 - self.epsilon)

    def weighted_majority_vote(self, predictions_dict):
        """
        根據各專家對同一問題的預測，做加權投票/加權合併：
        
        :param predictions_dict: dict, 形如
            {
                "expert_name1": "SQL1",
                "expert_name2": "SQL2",
                "expert_name3": "SQL1"
                ...
            }
        :return: (best_sql, chosen_experts)
            - best_sql: 權重最高的那條 SQL 預測
            - chosen_experts: list，哪些專家輸出此 best_sql
        """
        sql_to_weight = {}
        sql_to_experts = {}

        # 1. 整理相同 SQL 的累計權重
        for expert_name, sql_str in predictions_dict.items():
            w = self.experts.get(expert_name, 1.0)
            if sql_str not in sql_to_weight:
                sql_to_weight[sql_str] = 0.0
                sql_to_experts[sql_str] = []
            sql_to_weight[sql_str] += w
            sql_to_experts[sql_str].append(expert_name)

        # 2. 找出加總權重最大的那條 SQL
        best_sql = None
        best_weight = -1
        for sql_str, total_w in sql_to_weight.items():
            if total_w > best_weight:
                best_weight = total_w
                best_sql = sql_str

        # 3. 回傳該 best_sql 及其對應的專家清單
        chosen_experts = sql_to_experts[best_sql]
        return best_sql, chosen_experts

    def get_weights(self):
        """
        回傳目前所有專家的權重 (dict)。
        """
        return self.experts
