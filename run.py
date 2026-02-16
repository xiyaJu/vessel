import yaml
import pandas as pd
import numpy as np
from utils.vessel_trans import start_state
from matplotlib import animation
from matplotlib import pyplot as plt
from visualization.animate import animate_step
from utils.cpa_calculations import cpa_calculations
from utils.risk_calculations import risk_calculations

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run():
    config = load_config('config.yaml')

    t = config['initial_time']  # initial time
    dt = config['dt']  # time step
    Ts = dt  # sampling time
    N = round(config['sim_time'] / dt)  # number of steps
    num_ob = config['num_obstacles']

    Animation = config['animation']

    LOA_own, BOL_own = config['LOA_own'], config['BOL_own']
    CPA_own = LOA_own * 2

    vessel_data = pd.read_csv(config['vessel_data'])
    # initialize conditions
    x_v, y_v, psi_v ,u_v = start_state(vessel_data,0,config)     # 初始位置，航向（x正方向逆时针的夹角）,速度（米/秒）
    r_v, b_v = 0.0, 0.0  # 初始角速度、横漂速度
    ui_psi1 = 0.0  # 航向误差积分项（控制器用）

    X_0 = np.array([x_v, y_v, psi_v, r_v, b_v, u_v])
    X = X_0.copy()
    
    Xroute = [0, x_v+u_v*np.cos(psi_v)*config['sim_time']] # 原定轨迹
    Yroute = [0, y_v+u_v*np.sin(psi_v)*config['sim_time']]

    # obstacle data 换成多个障碍物的时候记得修改这里的循环
    X_ob, Y_ob, psi_ob ,u_ob = np.zeros(num_ob), np.zeros(num_ob), np.zeros(num_ob), np.zeros(num_ob)
    for i in range(num_ob):
        X_ob[i], Y_ob[i], psi_ob[i], u_ob[i] = start_state(vessel_data,1,config)

    # 初始化仿真记录数组（存储每一步的状态/指标）
    time = []  # 时间序列
    Kdir = np.ones(N)  # 转向系数（初始全1）
    x, y, psi = np.zeros(N), np.zeros(N), np.zeros(N)  # 本船位置/航向
    r, b, u = np.zeros(N), np.zeros(N), np.zeros(N)  # 角速度/横漂/纵向速度
    v_c = np.zeros(N)  # 控制速度
    u_p, tau_c, tau_ac = np.zeros(N), np.zeros(N), np.zeros(N)  # 速度指令/控制力矩/执行器输出力矩
    psi_p, psi_wp, psi_oa = np.zeros(N), np.zeros(N), np.zeros(N)  # 最终航向指令/航点航向/避碰航向
    V_x, V_y = np.zeros(N), np.zeros(N)  # 本船x/y方向速度

    # 障碍物状态数组（N步×障碍物数量）
    Xobs, Yobs, Vxobs, Vyobs = (np.zeros((N, num_ob)) for _ in range(4))
    LOA_ob = [config['LOA_ob']] * num_ob
    BOL_ob = [config['BOL_ob']] * num_ob
    CPA_ob = [LOA_ob * 2] * num_ob
    # CPA相关指标（DCPA/TCPA/相对速度/角度）
    DCPA, TCPA, Vrel, alpha, psi_Vrel = (np.zeros((N, num_ob)) for i in range(5))
    DCPA[:1], TCPA[:1] = 1000, 1000  # 初始值（避免除零）
    DCPA2, TCPA2, Vrel2, alpha2, psi_Vrel2 = (np.zeros((N, num_ob)) for _ in range(5))  # 静态CPA指标
    Distance_ob, Bearing_ob, Risk = (np.zeros((N, num_ob)) for _ in range(3))  # 距离/方位/碰撞风险

    Xobs[:, 0] = X_ob
    Yobs[:, 0] = Y_ob
    Vxobs[:, 0] = u_ob * np.cos(psi_ob)
    Vyobs[:, 0] = u_ob * np.sin(psi_ob)

    if Animation:
        fig, ax = plt.subplots()
        plt.plot(Xroute, Yroute, 'ob', Xroute, Yroute, ':b', linewidth=1.0)
        plt.grid(True)
        writer = animation.PillowWriter(fps=5)

    with writer.saving(fig, f"{config['output_dir']}/scenario_animation{config['case_number']}.gif", dpi=200):
        for i in range(N):
            # Record current state
            time.append(t)
            x[i] = X[0]
            y[i] = X[1]
            psi[i] = X[2]
            r[i] = X[3]
            b[i] = X[4]
            u[i] = X[5]
            X_0 = X.copy()

            # Speed command
            u_p[i] = 43.3

            X=X*0.9

            for j in range(num_ob):
                X_ob[j], Y_ob[j], psi_ob[j], u_ob[j] = start_state(vessel_data,2*i+1,config)
                Xobs[i, j] = X_ob[j]
                Yobs[i, j] = Y_ob[j]
                Vxobs[i, j] = u_ob[j] * np.cos(psi_ob[j])
                Vyobs[i, j] = u_ob[j] * np.sin(psi_ob[j])

            for j in range(num_ob):
                if i >= 1:
                    Distance_ob[i, j] = np.sqrt(
                        (np.array(Xobs[i, j]) - x[i])**2 + 
                        (np.array(Yobs[i, j]) - y[i])**2)
                    
                    DCPA[i, j], TCPA[i, j], Vrel[i, j], alpha[i, j], psi_Vrel[i, j] = cpa_calculations(
                        x[i], y[i], x[i-1], y[i-1], Xobs[i, j], Yobs[i, j], 
                        Xobs[i-1, j], Yobs[i-1, j], Ts
                    )

                Risk[i, j] = risk_calculations(
                    DCPA[i, j], TCPA[i, j], Distance_ob[i, j], Vrel[i, j])

            # Animation
            l = len(Risk[i, :])
            animate_step(
                x[i], y[i], psi[i],           # 本船位置和航向
                LOA_own, BOL_own, CPA_own,    # 本船尺寸和CPA圈
                Xobs, Yobs, psi_ob,              # 障碍物位置和航向
                LOA_ob, BOL_ob, CPA_ob,       # 障碍物尺寸和CPA圈
                Risk[i, :], u_ob, i, l                 # 风险值、速度、步数
            )
            
            if i % 101 == 0 and i != 0:
                writer.grab_frame()

            t += dt

        # Save animation plots
        plt.title(f'Case {config["case_number"]}', fontsize=25)
        plt.savefig(f'{config["output_dir"]}/simulation_result{config["case_number"]}.eps', format='eps')
        plt.savefig(f'{config["output_dir"]}/simulation_result{config["case_number"]}.png')
        plt.show(block=True)
       


if __name__ == '__main__':
    run()