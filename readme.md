# 项目说明
## 运行规则
- 本船：以数据集中第一行数据为起始状态，后续状态由模拟器控制
- 障碍船：直接读取数据集偶数行（第二条船的运行轨迹）

## 执行命令
```bash
python run.py
```

## 项目结构
```
├── config.yaml     
├── controllers     # 规划模块所有的代码
│   ├── controller.py   # 入口
│   ├── decision_maker.py
│   ├── env.py
│   ├── llm_agent_action.py     # 决策模块
│   ├── llm_agent_negotiation_system.py     # 协商模块
│   ├── llm_client.py
│   └── llm_config.yaml
├── data
│   └── path_data_1.csv
├── readme.md
├── results
│   ├── heading_over_time1.png      # 方向变化图
│   ├── scenario_animation_crsah.gif
│   ├── scenario_animation_origin.gif   # 不做任何决策时的轨迹
│   ├── scenario_animation1.gif     # 使用llm控制器后的轨迹
│   └── speed_over_time1.png    # 速度变化图
├── run.py      # 主函数
├── utils
│   ├── cpa_calculations.py
│   ├── risk_calculations.py
│   ├── vessel_trans.py
│   └── zmf.py
└── visualization   # 绘制gif图
    ├── animate.py
    ├── rendering.py
    └── save_animation.py
```