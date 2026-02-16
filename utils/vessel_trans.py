import math
import pandas as pd

def latlon_to_xy(lon: float, lat: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    """
    将经纬度转换为以指定原点为中心的笛卡尔坐标（单位：米）
    """
    R = 6371000  # 地球平均半径（米）
    # 转换为弧度
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    origin_lon_rad = math.radians(origin_lon)
    origin_lat_rad = math.radians(origin_lat)
    
    # 计算东西方向距离（x轴）
    dx = R * (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad)
    # 计算南北方向距离（y轴）
    dy = R * (lat_rad - origin_lat_rad)
    
    return dx, dy

def course_to_theta(course_deg: float) -> float:
    """
    将船舶航向（从正北顺时针的角度）转换为与x轴正方向（东）的逆时针夹角
    """
    theta_deg = 90 - course_deg  # 转换为从x轴正方向逆时针的角度
    if theta_deg < 0:
        theta_deg += 360  # 处理负角度
    return math.radians(theta_deg)  # 转换为弧度

def speed_to_mps(speed_knots: float) -> float:
    """
    将速度从节（knots）转换为米/秒（m/s）
    """
    return speed_knots * 0.514444

def start_state(vessel_data: pd.DataFrame, i: int, config: dict) -> tuple[float, float, float, float]:
    """
    从船舶轨迹数据中提取第i行的初始状态
    output:
        (x, y, theta, speed_mps):
            x: 笛卡尔x坐标（米，东为正）
            y: 笛卡尔y坐标（米，北为正）
            theta: 与x轴正方向的夹角（弧度，逆时针为正）
            speed_mps: 速度（米/秒）
    """
    origin_lon = config['origin_lon']
    origin_lat = config['origin_lat']
    # 提取第i行数据
    row = vessel_data.iloc[i]
    
    # 提取核心字段（根据列顺序：ID(0), 经度(1), 纬度(2), 速度(3), 航向(4)）
    lon = row.iloc[1]
    lat = row.iloc[2]
    speed = row.iloc[3]
    course = row.iloc[4]
    
    # 经纬度转笛卡尔坐标
    x, y = latlon_to_xy(lon, lat, origin_lon, origin_lat)
    
    # 航向角转换
    theta = course_to_theta(course)
    
    # 速度单位转换
    speed_mps = speed_to_mps(speed)
    
    return x, y, theta, speed_mps