import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from .llm_client import QwenAPIClient

class LlmAgentActionModule:
    def __init__(self, env):
        """
        :param env: ShipCollisionEnv实例（包含配置+数据）
        """
        self.env = env
        self.action_results = {}
        # 初始化千问客户端
        self.qwen_client = QwenAPIClient(
            api_key=self.env.llm_config["api_key"],
            model_name=self.env.llm_config["model_name"],
            base_url=self.env.llm_config["base_url"]
        )

    def generate_action_prompt(self, conflict_pair: Dict, priority_result: Dict) -> str:
        """生成动作决策的LLM提示词"""
        own_ship_state = self.env.get_own_ship_state()
        if priority_result["priority"] == "no_action":
            return f"""
            障碍船{conflict_pair['obs_index']}无碰撞风险，本船保持当前航向{np.degrees(own_ship_state[2]):.1f}°、速度{own_ship_state[3]:.2f}m/s不变。
            仅返回JSON：{{"u_cmd": "keep", "psi_cmd": "keep"}}
            """.strip()
        
        return f"""
        基于优先级协商结果（{priority_result['reason']}），为避免与障碍船{conflict_pair['obs_index']}碰撞，
        请为本船生成具体的动作指令，严格遵守以下要求：
        1. 速度指令仅可选：'faster'（加速）、'slower'（减速）、'keep'（保持）；
        2. 航向指令仅可选：'turn_left'（左转）、'turn_right'（右转）、'keep'（保持）；
        3. 必须输出JSON格式，示例：{{"u_cmd": "keep", "psi_cmd": "turn_right", "reason": "阐述原因"}}；
        4. 动作需符合COLREGs规则，幅度合理，避免极端操作。
        5. 尽量留足充足的时间和空间，DCPA至少需要预留100m，TCPA至少需要预留300s
        6. 当本船基于COLREGs规则为直航船，但障碍船无明显避让动作时，本船应主动采取避让行为。
        7. 请尽可能避免航船速度降至2m/s以下，每次加速或减速会造成速度变化0.2m/s；请尽量避免航船方向变化过大，每次改变方向会产生0.1°的航向变化。
        
        本船当前状态：航向{np.degrees(own_ship_state[2]):.1f}°，速度{own_ship_state[3]:.2f}m/s；
        障碍船状态：航向{conflict_pair['obstacle_ship']['heading_deg']:.1f}°，速度{conflict_pair['obstacle_ship']['speed']:.2f}m/s；
        碰撞风险指标：DCPA={conflict_pair['collision_metrics']['dcpa']:.2f}m，TCPA={conflict_pair['collision_metrics']['tcpa']:.2f}s。
        """.strip()

    def generate_action_command(self, conflict_pairs: List[Dict], priority_results: Dict) -> Tuple[Optional[str], Optional[str]]:
        """
        调用千问API生成动作指令
        :param conflict_pairs: 协商模块返回的冲突对列表（修复核心：新增该参数）
        :param priority_results: 优先级协商结果
        """
        # 默认动作：保持
        final_u_cmd = "keep"
        final_psi_cmd = "keep"
        # 筛选有冲突的障碍船（从传入的conflict_pairs读取，而非env）
        conflict_obs_indices = [p["obs_index"] for p in conflict_pairs if p["is_conflict"]]

        if not conflict_obs_indices:
            return None, None

        # 遍历冲突船生成动作
        for obs_idx in conflict_obs_indices:
            conflict_pair = next(p for p in conflict_pairs if p["obs_index"] == obs_idx)
            priority_result = priority_results[obs_idx]

            # 生成提示词并调用API
            action_prompt = self.generate_action_prompt(conflict_pair, priority_result)
            print(f"action_prompt: {action_prompt}")
            llm_response = self.qwen_client.call_qwen(
                system_prompt=self.env.prompt_config["action_system"],
                user_prompt=action_prompt,
                temperature=self.env.llm_config["temperature"],
                max_tokens=self.env.llm_config["max_tokens"]
            )
            print(f"llm_action_response: {llm_response}")
            # 解析动作指令
            if not llm_response:
                self.action_results[obs_idx] = {
                    "u_cmd": "keep", "psi_cmd": "keep", "error": "LLM API调用失败"
                }
                continue

            try:
                # 提取JSON内容
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if not json_match:
                    raise ValueError("未提取到JSON动作指令")
                
                action_dict = json.loads(json_match.group())
                # 校验指令合法性
                valid_u_cmds = ["faster", "slower", "keep"]
                valid_psi_cmds = ["turn_left", "turn_right", "keep"]
                
                action_dict["u_cmd"] = action_dict.get("u_cmd", "keep")
                action_dict["psi_cmd"] = action_dict.get("psi_cmd", "keep")
                
                if action_dict["u_cmd"] not in valid_u_cmds:
                    action_dict["u_cmd"] = "keep"
                if action_dict["psi_cmd"] not in valid_psi_cmds:
                    action_dict["psi_cmd"] = "keep"
                
                self.action_results[obs_idx] = action_dict
                # 多船场景：取最后一个冲突船的动作（可扩展为加权融合）
                final_u_cmd = action_dict["u_cmd"]
                final_psi_cmd = action_dict["psi_cmd"]
            except Exception as e:
                print(f"解析动作指令失败（障碍船{obs_idx}）：{e}")
                self.action_results[obs_idx] = {
                    "u_cmd": "keep", "psi_cmd": "keep", "error": str(e)
                }

        # 转换为Controller需要的格式（keep→None）
        final_u_cmd = final_u_cmd if final_u_cmd != "keep" else None
        final_psi_cmd = final_psi_cmd if final_psi_cmd != "keep" else None

        return final_u_cmd, final_psi_cmd