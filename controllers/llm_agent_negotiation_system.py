import re
import json
import numpy as np
from typing import Dict, List, Tuple
from .llm_client import QwenAPIClient
from .env import wrap_to_pi

class LlmAgentNegotiationModule:
    """LLM智能体-冲突协商模块"""
    def __init__(self, env):
        """
        :param env: ShipCollisionEnv实例（包含配置+数据）
        """
        self.env = env
        self.conflict_pairs = []
        self.priority_results = {}
        # 初始化千问客户端（从Env读取配置）
        self.qwen_client = QwenAPIClient(
            api_key=self.env.llm_config["api_key"],
            model_name=self.env.llm_config["model_name"],
            base_url=self.env.llm_config["base_url"]
        )

    def detect_conflict_pairs(self) -> List[Dict]:
        """检测冲突对（从Env读取数据）"""
        self.conflict_pairs = []
        own_ship_state = self.env.get_own_ship_state()
        obs_states = self.env.get_obs_state_list()
        own_x, own_y, own_psi, own_u = own_ship_state

        for obs_idx in range(self.env.num_obstacles):
            obs_x, obs_y, obs_psi, obs_u = obs_states[obs_idx]
            # 冲突判定：风险阈值 + 有效TCPA + 距离限制
            risk_threshold = self.env.conflict_config["risk_threshold"]
            max_distance = self.env.conflict_config["max_distance"]
            # if (self.env.risk[obs_idx] > risk_threshold and 
            #     self.env.tcpa[obs_idx] >= 0 and 
            #     self.env.distance_ob[obs_idx] < max_distance):
            if (self.env.tcpa[obs_idx] >= 0):
                # 构造冲突对信息
                metrics = self.env.get_conflict_metrics(obs_idx)
                self.conflict_pairs.append({
                    "obs_index": obs_idx,
                    "own_ship": {
                        "x": own_x, "y": own_y, "heading": own_psi, "speed": own_u,
                        "heading_deg": np.degrees(own_psi)
                    },
                    "obstacle_ship": {
                        "x": obs_x, "y": obs_y, "heading": obs_psi, "speed": obs_u,
                        "heading_deg": np.degrees(obs_psi)
                    },
                    "collision_metrics": metrics,
                    "is_conflict": True
                })
            else:
                self.conflict_pairs.append({
                    "obs_index": obs_idx,
                    "is_conflict": False
                })
        return self.conflict_pairs

    def generate_conflict_description(self, conflict_pair: Dict) -> str:
        """生成冲突对的自然语言描述"""
        if not conflict_pair["is_conflict"]:
            return f"障碍船{conflict_pair['obs_index']}：无碰撞风险，无需协商优先级。"
        
        own_info = conflict_pair["own_ship"]
        obs_info = conflict_pair["obstacle_ship"]
        metrics = conflict_pair["collision_metrics"]
        
        return f"""
        COLREGs核心规则：
        Rule 13：追越局面；一船从他船正横后大于22.5°方向驶近并赶上他船；追越船为让路船，被追越船为直航船；
        Rule 14：对遇局面；两机动船在相反或接近相反航向相遇，有碰撞危险；双方均为让路船，无直航船；
        Rule 15：交叉相遇局面；两机动船交叉航向，有碰撞危险（非对遇、非追越）；有他船在本船右舷为让路船，他船在本船左舷为直航船；

        冲突对信息（本船 vs 障碍船{conflict_pair['obs_index']}）：
        1. 本船状态：位置({own_info['x']:.2f}m, {own_info['y']:.2f}m)，航向{own_info['heading_deg']:.1f}°，速度{own_info['speed']:.2f}m/s；
        2. 障碍船状态：位置({obs_info['x']:.2f}m, {obs_info['y']:.2f}m)，航向{obs_info['heading_deg']:.1f}°，速度{obs_info['speed']:.2f}m/s；
        3. 碰撞风险指标：相对方位{metrics['bearing_deg']:.1f}°，最近会遇距离(DCPA){metrics['dcpa']:.2f}m，最近会遇时间(TCPA){metrics['tcpa']:.2f}s，碰撞风险值{metrics['risk']:.2f}；
        请你严格按照下面的格式来回答
        {{
            "obs_index": 0,
            "priority": "",
            "reason": "",
            "rule": "阐述具体属于哪条COLREGs规则"
        }}
        """.strip()

    def negotiate_priority(self) -> Dict:
        """调用千问API协商优先级"""
        self.priority_results = {}
        # 生成所有冲突对描述
        all_descriptions = [self.generate_conflict_description(pair) for pair in self.conflict_pairs]
        user_prompt = "\n\n".join(all_descriptions)
        # print(f"user_prompt: {user_prompt}")
        # 调用千问API
        llm_response = self.qwen_client.call_qwen(
            system_prompt=self.env.prompt_config["negotiation_system"],
            user_prompt=user_prompt,
            temperature=self.env.llm_config["temperature"],
            max_tokens=self.env.llm_config["max_tokens"]
        )
        print(f"llm_response: {llm_response}")

        # 处理API返回结果
        if not llm_response:
            # API调用失败：返回默认值
            for pair in self.conflict_pairs:
                self.priority_results[pair["obs_index"]] = {
                    "priority": "no_action",
                    "reason": "LLM API调用失败，默认无动作"
                }
            return self.priority_results

        # 解析JSON结果（兼容LLM返回的多余文本）
        try:
            # 提取JSON字符串（支持列表/单个对象）
            json_match = re.search(r'\[.*\]|\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("未提取到JSON内容")
            
            json_str = json_match.group()
            llm_results = json.loads(json_str)
            
            # 适配单/多冲突对格式
            if isinstance(llm_results, dict):
                llm_results = [llm_results]
            
            # 映射到优先级字典
            for result in llm_results:
                obs_idx = result.get("obs_index", -1)
                if 0 <= obs_idx < self.env.num_obstacles:
                    self.priority_results[obs_idx] = result
            
            # 补充未返回的冲突对
            for pair in self.conflict_pairs:
                obs_idx = pair["obs_index"]
                if obs_idx not in self.priority_results:
                    self.priority_results[obs_idx] = {
                        "priority": "no_action",
                        "reason": "未获取到LLM优先级判定结果"
                    }
        except Exception as e:
            print(f"解析LLM优先级结果失败：{e}")
            for pair in self.conflict_pairs:
                self.priority_results[pair["obs_index"]] = {
                    "priority": "no_action",
                    "reason": f"解析失败：{str(e)}"
                }

        return self.priority_results