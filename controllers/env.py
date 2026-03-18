import yaml
import numpy as np
from typing import Tuple, List, Dict, Optional

# 角度归一化函数
def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

class ShipCollisionEnv:
    """
    船舶避碰环境类：
    1. 封装Controller的所有输入数据
    2. 加载并管理LLM/冲突检测配置
    """
    def __init__(
        self,
        # 本船状态
        own_ship_x: float,
        own_ship_y: float,
        own_ship_psi: float,
        own_ship_u: float,
        # 障碍船状态
        obs_x: np.ndarray,
        obs_y: np.ndarray,
        obs_psi: np.ndarray,
        obs_u: np.ndarray,
        # 碰撞风险指标
        bearing_ob: np.ndarray,
        dcpa: np.ndarray,
        tcpa: np.ndarray,
        risk: np.ndarray,
        distance_ob: np.ndarray,
        # 配置文件路径（可选）
        config_path: str = "controllers/llm_config.yaml"
    ):
        # 1. 加载配置文件
        self.config = self._load_config(config_path)
        self.llm_config = self.config["qwen"]
        self.prompt_config = self.config["prompts"]
        self.conflict_config = self.config["conflict"]

        # 2. 本船状态（自动归一化航向）
        self.own_ship_x = own_ship_x
        self.own_ship_y = own_ship_y
        self.own_ship_psi = wrap_to_pi(own_ship_psi)
        self.own_ship_u = own_ship_u

        # 3. 障碍船状态（批量归一化航向）
        self.obs_x = obs_x
        self.obs_y = obs_y
        self.obs_psi = np.vectorize(wrap_to_pi)(obs_psi)
        self.obs_u = obs_u
        self.num_obstacles = len(obs_x)

        # 4. 碰撞风险指标
        self.bearing_ob = bearing_ob
        self.dcpa = dcpa
        self.tcpa = tcpa
        self.risk = risk
        self.distance_ob = distance_ob

        # 5. 数据校验
        self._validate_data()

    def _load_config(self, config_path: str) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"加载配置文件失败：{e}")

    def _validate_data(self) -> None:
        """校验障碍船数据维度一致性"""
        obs_arrays = [
            self.obs_x, self.obs_y, self.obs_psi, self.obs_u,
            self.bearing_ob, self.dcpa, self.tcpa, self.risk, self.distance_ob
        ]
        for arr in obs_arrays:
            if len(arr) != self.num_obstacles:
                raise ValueError(
                    f"障碍船数据维度不匹配！期望{self.num_obstacles}个，实际{len(arr)}个"
                )

    # 便捷方法：获取结构化状态
    def get_own_ship_state(self) -> Tuple[float, float, float, float]:
        """返回本船状态元组"""
        return (self.own_ship_x, self.own_ship_y, self.own_ship_psi, self.own_ship_u)

    def get_obs_state_list(self) -> List[Tuple[float, float, float, float]]:
        """返回障碍船状态列表"""
        return [
            (self.obs_x[i], self.obs_y[i], self.obs_psi[i], self.obs_u[i])
            for i in range(self.num_obstacles)
        ]

    def get_conflict_metrics(self, obs_index: int) -> Dict:
        """获取指定障碍船的碰撞指标（格式化）"""
        if obs_index < 0 or obs_index >= self.num_obstacles:
            raise IndexError(f"障碍船索引{obs_index}超出范围（0~{self.num_obstacles-1}）")
        return {
            "bearing_deg": np.degrees(wrap_to_pi(self.bearing_ob[obs_index])),
            "dcpa": self.dcpa[obs_index],
            "tcpa": self.tcpa[obs_index],
            "risk": self.risk[obs_index],
            "distance": self.distance_ob[obs_index]
        }

    def __repr__(self) -> str:
        """自定义打印格式"""
        return (
            f"ShipCollisionEnv(本船位置=({self.own_ship_x:.2f}, {self.own_ship_y:.2f}), "
            f"航向={np.degrees(self.own_ship_psi):.1f}°, 速度={self.own_ship_u:.2f}m/s, "
            f"障碍船数量={self.num_obstacles})"
        )