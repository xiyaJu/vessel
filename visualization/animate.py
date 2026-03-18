import numpy as np
from .rendering import animate_ship, animate_static_obstacle

def _fmt(v, fmt="{:.2f}"):
    """把 nan/inf 安全格式化"""
    if v is None or (isinstance(v, float) and (not np.isfinite(v))):
        return "N/A"
    return fmt.format(v)

def animate_step(
    ax,
    info_text,
    t,
    x, y, psi,
    LOA_own, BOL_own, CPA_own,
    x_obs, y_obs, psi_obs, u_obs,
    LOA_ob, BOL_ob, CPA_ob,
    Bearing, DCPA, TCPA, Risk, Distance
):
    """
    每次调用画“当前这一帧”，并更新右侧信息栏。
    注意：帧率由主循环 frame_stride 控制，这里不再 step%100 限制。
    """
    #(f"[DEBUG] own: ({x:.1f}, {y:.1f}), obs[0]: ({x_obs[0]:.1f}, {y_obs[0]:.1f})")
    # 清掉上一帧的船/障碍物 注释掉可以显示轨迹
    for p in list(ax.patches):
        p.remove()

    #print(f"[SHIP VERTICES] x={x:.1f}, y={y:.1f}, psi={psi:.3f}, loa={LOA_own*3}, bol={BOL_own*3}")

    # 画本船
    animate_ship(ax, x, y, psi, LOA_own, BOL_own, CPA_own, color=[0.41, 0.0, 0.41])

    #print(f"[PATCH DEBUG] patches count: {len(ax.patches)}, xlim: {ax.get_xlim()}, ylim: {ax.get_ylim()}")
    # 画障碍物
    colors = [
        [0, 0, 1],      # Blue
        [1, 0.5, 0],    # Orange
        [0, 1, 0],      # Green
        [0.7, 0, 0.7],  # Purple
        [0, 0.7, 0.7],  # Cyan-ish
    ]

    for j in range(len(u_obs)):
        obs_col = [0.0, 0.7, 0.0]
        r = Risk[j]
        if np.isfinite(r):
            if r > 0.75:
                obs_col = [1.0, 0.0, 0.0]
            elif r > 0.6:
                obs_col = [1.0, 0.6, 0.0]
            elif r > 0.35:
                obs_col = [1.0, 0.9, 0.0]

        if u_obs[j] > 0.5:
            animate_ship(
                ax,
                x_obs[j], y_obs[j], psi_obs[j],
                LOA_ob[j], BOL_ob[j], CPA_ob[j],
                color=colors[j % len(colors)]
            )
        else:
            animate_static_obstacle(ax, x_obs[j], y_obs[j], CPA_ob[j], obs_col)
    
    ax.set_xlim(-3500, 3500)
    ax.set_ylim(-3500, 3500)
    # ====== 更新右侧信息栏 ======
    lines = [f"t = {t:.2f} s"]
    for j in range(len(u_obs)):
        # Bearing(rad) -> deg
        bearing_deg = np.degrees(Bearing[j]) if np.isfinite(Bearing[j]) else np.nan

        lines += [
            f"obstacle: {j}",
            f"  Bearing:  {_fmt(bearing_deg, '{:.1f}')} deg",
            f"  DCPA:     {_fmt(DCPA[j], '{:.1f}')} m",
            f"  TCPA:     {_fmt(TCPA[j], '{:.1f}')} s",
            f"  Risk:     {_fmt(Risk[j], '{:.2f}')}",
            f"  Distance: {_fmt(Distance[j], '{:.1f}')} m",
        ]

    info_text.set_text("\n".join(lines))
