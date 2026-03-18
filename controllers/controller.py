from typing import Tuple, Optional
import numpy as np
from .env import ShipCollisionEnv
from .llm_agent_negotiation_system import LlmAgentNegotiationModule
from .llm_agent_action import LlmAgentActionModule

def ShipCollisionController(
    x: float, y: float, psi: float, u: float,
    Xobs: np.ndarray, Yobs: np.ndarray, Psiobs: np.ndarray, Uobs: np.ndarray,
    Bearing_ob: np.ndarray, DCPA: np.ndarray, TCPA: np.ndarray, Risk: np.ndarray, Distance_ob: np.ndarray,
    config_path: str = "controllers/llm_config.yaml"
) -> Tuple[Optional[str], Optional[str]]:
    """
    船舶避碰控制器（对接原有仿真代码）
    :return: (u_cmd, psi_cmd) 速度指令/航向指令
    """
    try:
        # 1. 初始化Env（自动加载配置+校验数据）
        env = ShipCollisionEnv(
            own_ship_x=x, own_ship_y=y, own_ship_psi=psi, own_ship_u=u,
            obs_x=Xobs, obs_y=Yobs, obs_psi=Psiobs, obs_u=Uobs,
            bearing_ob=Bearing_ob, dcpa=DCPA, tcpa=TCPA, risk=Risk, distance_ob=Distance_ob,
            config_path=config_path
        )

        # 2. 初始化LLM模块（直接传入Env）
        negotiation_module = LlmAgentNegotiationModule(env)
        action_module = LlmAgentActionModule(env)

        # 3. 冲突检测 + 优先级协商 + 动作生成
        conflict_pairs = negotiation_module.detect_conflict_pairs()
        print(f"conflict_pairs: {conflict_pairs}")
        priority_results = negotiation_module.negotiate_priority()
        u_cmd, psi_cmd = action_module.generate_action_command(conflict_pairs,priority_results)

        return u_cmd, psi_cmd
    except Exception as e:
        print(f"控制器执行失败：{e}")
        # 异常时返回None（保持原有状态）
        return None, None