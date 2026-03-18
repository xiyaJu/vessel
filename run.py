import yaml
import pandas as pd
import numpy as np
from utils.vessel_trans import start_state
from matplotlib import animation
from matplotlib import pyplot as plt
from visualization.animate import animate_step
from utils.cpa_calculations import cpa_calculations
from utils.risk_calculations import risk_calculations

from controllers.controller import ShipCollisionController
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
# 更新状态（包括位置、航向、速度）
def update_state(S, u_cmd, psi_cmd, dt):
    """
    Update the vessel state based on the control commands.
    """
    x_v, y_v, psi_v, u_v = S
    if u_cmd=="faster":
        u_v += 0.2
    elif u_cmd=="slower":
        u_v -= 0.2
    elif u_cmd=="keep":
        u_v = u_v
    
    if psi_cmd=="turn_left":
        psi_v += 0.1
        psi_v = wrap_to_pi(psi_v)
    elif psi_cmd=="turn_right":
        psi_v -= 0.1
        psi_v = wrap_to_pi(psi_v)
    elif psi_cmd=="keep":
        psi_v = psi_v
    x_v += u_v * np.cos(psi_v) * dt
    y_v += u_v * np.sin(psi_v) * dt
    S = np.array([x_v, y_v, psi_v, u_v])
    return S

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
    u_v=2.0

    S = np.array([x_v, y_v, psi_v, u_v])

    # obstacle data 多个障碍物
    x_ob, y_ob, psi_ob ,u_ob = np.zeros(num_ob), np.zeros(num_ob), np.zeros(num_ob), np.zeros(num_ob)

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
    Xobs, Yobs = (np.zeros((N, num_ob)) for _ in range(2))
    Psiobs, Uobs = (np.zeros((N, num_ob)) for _ in range(2))
    LOA_ob = [config['LOA_ob']] * num_ob
    BOL_ob = [config['BOL_ob']] * num_ob
    CPA_ob = [loa * 2 for loa in LOA_ob]
    # CPA相关指标（DCPA/TCPA/相对速度/角度）
    DCPA, TCPA, Vrel, alpha, psi_Vrel = (np.zeros((N, num_ob)) for i in range(5))
    DCPA[:1], TCPA[:1] = 1000, 1000  # 初始值（避免除零）
    DCPA2, TCPA2, Vrel2, alpha2, psi_Vrel2 = (np.zeros((N, num_ob)) for _ in range(5))  # 静态CPA指标
    Distance_ob, Bearing_ob, Risk = (np.zeros((N, num_ob)) for _ in range(3))  # 距离/方位/碰撞风险

    if Animation:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.grid(True)

        # 坐标轴范围固定，避免缩放
        ax.set_xlim(-3500, 3500)
        ax.set_ylim(-3500, 3500)
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale(False)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # 右侧留白用于侧栏
        fig.subplots_adjust(right=0.72)

        # 右侧信息栏 text（每帧更新）
        info_text = ax.text(
            1.02, 0.98, "", transform=ax.transAxes,
            va="top", ha="left", clip_on=False,
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6")
        )

        writer = animation.PillowWriter(fps=5)

        # 用 stride 控制“每隔多少步抓一帧”
        frame_stride = config['gif_stride']


    with writer.saving(fig, f"{config['output_dir']}/scenario_animation{config['case_number']}.gif", dpi=200):
        for i in range(N):
            # Record current state
            time.append(t)
            x[i] = S[0]
            y[i] = S[1]
            psi[i] = S[2]
            u[i] = S[3]
            # 更新障碍物状态
            for j in range(num_ob):
                x_ob[j], y_ob[j], psi_ob[j], u_ob[j] = start_state(vessel_data,2*i*dt+1,config)

                # 记录状态
                Xobs[i, j] = x_ob[j]
                Yobs[i, j] = y_ob[j]
                Psiobs[i, j] = psi_ob[j]
                Uobs[i, j] = u_ob[j]

                dx = Xobs[i, j] - x[i]
                dy = Yobs[i, j] - y[i]
                Distance_ob[i, j] = np.hypot(dx, dy)

                bearing_true = np.arctan2(dy, dx)
                Bearing_ob[i, j] = wrap_to_pi(bearing_true - psi[i])
                
                if i > 0:
                    DCPA[i, j], TCPA[i, j], Vrel[i, j], alpha[i, j], psi_Vrel[i, j] = cpa_calculations(
                        x[i], y[i], x[i-1], y[i-1], Xobs[i, j], Yobs[i, j], 
                        Xobs[i-1, j], Yobs[i-1, j], Ts
                    )

                    Risk[i, j] = risk_calculations(
                        DCPA[i, j], TCPA[i, j], Distance_ob[i, j], Vrel[i, j])
                    
            

            if Animation and (i % frame_stride == 0):
                animate_step(
                    ax=ax,
                    info_text=info_text,
                    t=t,
                    x=x[i], y=y[i], psi=psi[i],
                    LOA_own=LOA_own, BOL_own=BOL_own, CPA_own=CPA_own,
                    x_obs=Xobs[i, :], y_obs=Yobs[i, :], psi_obs=Psiobs[i, :], u_obs=Uobs[i, :],
                    LOA_ob=LOA_ob, BOL_ob=BOL_ob, CPA_ob=CPA_ob,
                    Bearing=Bearing_ob[i, :],
                    DCPA=DCPA[i, :],
                    TCPA=TCPA[i, :],
                    Risk=Risk[i, :],
                    Distance=Distance_ob[i, :]
                )
                writer.grab_frame()
            # 这里添加成实际的速度推理
            ######################
            if i>20:
                u_cmd, psi_cmd = ShipCollisionController(
                    x[i], y[i], psi[i], u[i],
                    Xobs[i, :], Yobs[i, :], Psiobs[i, :], Uobs[i, :],
                    Bearing_ob[i, :], DCPA[i, :], TCPA[i, :], Risk[i, :], Distance_ob[i, :]
                )
            else:
                u_cmd, psi_cmd = "keep","keep"
            # u_cmd, psi_cmd = "faster","keep"
            ######################
            print('i:', i,'S:', S)
            S = update_state(S, u_cmd, psi_cmd, dt)

            t += dt

    # 绘制航向变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time, psi, label='Ship Heading')
    plt.plot(time, Psiobs[:, 0], 'ro', label='Obstacle Heading')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (rad)')
    plt.legend()
    plt.title(f'Case {config["case_number"]} - Heading Over Time')
    plt.savefig(f'{config["output_dir"]}/heading_over_time{config["case_number"]}.png')
    plt.show(block=True)

    # 绘制速度变化图
    plt.figure(figsize=(10, 6))
    plt.plot(time, u, label='Ship Speed')
    plt.plot(time, Uobs[:, 0], 'ro', label='Obstacle Speed')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.title(f'Case {config["case_number"]} - Speed Over Time')
    plt.savefig(f'{config["output_dir"]}/speed_over_time{config["case_number"]}.png')
    plt.show(block=True)
       


if __name__ == '__main__':
    run()
