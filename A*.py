import heapq
import numpy as np
from pyproj import Transformer
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Set
from matplotlib.lines import Line2D

# ----------------------- A* 算法实现 -----------------------

@dataclass(order=True)
class PriorityNode:
    f: float
    position: Tuple[int, int, int] = field(compare=False)
    g: float = field(compare=False)
    h: float = field(compare=False)
    parent: Optional['PriorityNode'] = field(compare=False, default=None)

def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """
    启发式函数：欧几里得距离
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(node: PriorityNode, grid: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    获取当前节点的所有可行邻居（最多26个方向）
    """
    neighbors = []
    x, y, z = node.position
    # 三维邻居，包括26个方向
    directions = [
        (-1,  0,  0), (1,  0,  0),
        (0, -1,  0), (0,  1,  0),
        (0,  0, -1), (0,  0,  1),
        (-1, -1,  0), (-1, 1,  0),
        (1, -1,  0), (1, 1,  0),
        (-1, 0, -1), (-1, 0, 1),
        (1, 0, -1), (1, 0, 1),
        (0, -1, -1), (0, -1, 1),
        (0, 1, -1), (0, 1, 1),
        (-1, -1, -1), (-1, -1, 1),
        (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1),
        (1, 1, -1), (1, 1, 1)
    ]
    for dx, dy, dz in directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
            if grid[nx, ny, nz] == 0:  # 0 表示可通行，其他值表示障碍
                neighbors.append((nx, ny, nz))
    return neighbors

def a_star_3d(start: Tuple[int, int, int], goal: Tuple[int, int, int], grid: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
    """
    实现A*算法进行三维路径规划
    """
    open_set = []
    start_node = PriorityNode(f=heuristic(start, goal), position=start, g=0, h=heuristic(start, goal))
    heapq.heappush(open_set, start_node)
    
    open_set_hash: Set[Tuple[int, int, int]] = {start}
    closed_set: Set[Tuple[int, int, int]] = set()
    
    came_from: Dict[Tuple[int, int, int], PriorityNode] = {}
    
    while open_set:
        current_node = heapq.heappop(open_set)
        open_set_hash.remove(current_node.position)
        
        if current_node.position == goal:
            # 回溯路径
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # 反转路径
        
        closed_set.add(current_node.position)
        
        for neighbor_pos in get_neighbors(current_node, grid):
            if neighbor_pos in closed_set:
                continue
            
            tentative_g = current_node.g + heuristic(current_node.position, neighbor_pos)
            
            neighbor_node = came_from.get(neighbor_pos)
            if neighbor_node is None or tentative_g < neighbor_node.g:
                h = heuristic(neighbor_pos, goal)
                f = tentative_g + h
                neighbor_node = PriorityNode(f=f, position=neighbor_pos, g=tentative_g, h=h, parent=current_node)
                came_from[neighbor_pos] = neighbor_node
                if neighbor_pos not in open_set_hash:
                    heapq.heappush(open_set, neighbor_node)
                    open_set_hash.add(neighbor_pos)
    
    return None  # 无路径

# ----------------------- 坐标转换 -----------------------

def convert_geo_to_local(lat, lon, alt, home_easting, home_northing, transformer, offset_x, offset_y, grid_shape):
    """
    将地理坐标(lat, lon, alt)转换为本地坐标(x, y, z)。
    """
    try:
        easting, northing = transformer.transform(lon, lat)
        x = easting - home_easting
        y = northing - home_northing
        z = alt  # 如果需要相对高度，可以用 z = alt - home_alt
        # 添加偏移量以确保所有坐标为正
        local_x = int(round(x)) + offset_x
        local_y = int(round(y)) + offset_y
        local_z = int(round(z))
        # 检查是否超出网格范围
        if not (0 <= local_x < grid_shape[0] and 0 <= local_y < grid_shape[1] and 0 <= local_z < grid_shape[2]):
            raise ValueError(f"本地坐标超出网格范围: ({local_x}, {local_y}, {local_z})")
        return (local_x, local_y, local_z)
    except Exception as e:
        print(f"坐标转换错误: {e}")
        sys.exit(1)

def convert_local_to_geo(x, y, z, home_easting, home_northing, transformer, offset_x, offset_y):
    """
    将本地坐标(x, y, z)转换为地理坐标(lat, lon, alt)。
    """
    try:
        # 移除偏移量
        real_x = x - offset_x
        real_y = y - offset_y
        easting = home_easting + real_x
        northing = home_northing + real_y

        # 进行坐标转换
        lon, lat = transformer.transform(easting, northing)  # 移除了 direction 参数

        # 检查转换结果是否为有限数值
        if not (np.isfinite(lat) and np.isfinite(lon)):
            raise ValueError(f"转换结果无效: lat={lat}, lon={lon}")

        alt = z  # 如果需要相对高度，可以用 alt = z + home_alt
        return lat, lon, alt
    except Exception as e:
        print(f"坐标转换错误: {e}")
        sys.exit(1)

def convert_path_to_waypoints(path, home_easting, home_northing, transformer, offset_x, offset_y):
    """
    将路径转换为Mission Planner的航点格式。
    """
    waypoints = []
    for i, point in enumerate(path):
        x, y, z = point
        lat, lon, alt = convert_local_to_geo(x, y, z, home_easting, home_northing, transformer, offset_x, offset_y)
        waypoint = {
            'index': i,
            'current_wp': 1 if i == 0 else 0,  # 起始点设为当前航点
            'frame': 3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT (3)
            'command': 16,  # MAV_CMD_NAV_WAYPOINT
            'param1': 0,    # Hold time (seconds)
            'param2': 0,    # Acceptance radius (meters)
            'param3': 0,    # Pass radius (meters)
            'param4': 0,    # Yaw (degrees)
            'lat': lat,
            'lon': lon,
            'alt': alt,
            'autocontinue': 1  # 自动继续
        }
        waypoints.append(waypoint)
    return waypoints

# ----------------------- 航点文件生成函数 -----------------------

def write_waypoints_file(waypoints, filename='mission.waypoints'):
    """
    将航点列表写入Mission Planner的.waypoints文件。
    """
    try:
        with open(filename, 'w') as f:
            # 写入文件头
            f.write("QGC WPL 110\n")
            # 写入每个航点
            for wp in waypoints:
                line = (
                    f"{wp['index']}\t"
                    f"{wp['current_wp']}\t"
                    f"{wp['frame']}\t"
                    f"{wp['command']}\t"
                    f"{wp['param1']}\t"
                    f"{wp['param2']}\t"
                    f"{wp['param3']}\t"
                    f"{wp['param4']}\t"
                    f"{wp['lat']}\t"
                    f"{wp['lon']}\t"
                    f"{wp['alt']}\t"
                    f"{wp['autocontinue']}\n"
                )
                f.write(line)
        print(f"航点文件已生成：{filename}")
    except Exception as e:
        print(f"写入航点文件错误: {e}")
        sys.exit(1)

# ----------------------- 可视化函数（优化后） -----------------------

def visualize(grid, path=None, start=None, goal=None, elev=30, azim=45):
    """
    可视化三维网格中的障碍物和路径

    :param grid: 三维网格
    :param path: 路径列表，格式为[(x1, y1, z1), (x2, y2, z2), ...]
    :param start: 起点坐标 (x, y, z)
    :param goal: 终点坐标 (x, y, z)
    :param elev: 视角的仰角（degrees）
    :param azim: 视角的方位角（degrees）
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 定义障碍物类型对应的颜色
    obstacle_colors = {
        1: 'saddlebrown',    # 树干
        2: 'forestgreen',    # 树冠
        3: 'grey',           # 立方体
        4: 'red',            # 球形
        # 添加更多类型时在此处扩展
    }

    # 绘制不同类型的障碍物
    for obstacle_type, color in obstacle_colors.items():
        obstacle_points = np.argwhere(grid == obstacle_type)
        if obstacle_points.size > 0:
            ax.scatter(obstacle_points[:,0], obstacle_points[:,1], obstacle_points[:,2],
                       c=color, marker='s', alpha=0.3)  # 降低透明度

    # 绘制路径
    if path:
        path = np.array(path)
        ax.plot(path[:,0], path[:,1], path[:,2], color='blue', linewidth=3, label='Path', zorder=5)

    # 绘制起点和终点
    if start:
        ax.scatter(*start, c='green', marker='o', s=100, label='Start', zorder=6)
    if goal:
        ax.scatter(*goal, c='purple', marker='x', s=100, label='Goal', zorder=6)

    # 设置坐标轴标签
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')

    # 设置标题
    ax.set_title('3D Path Planning using A* Algorithm')

    # 创建一个统一的障碍物图例
    proxy_obstacle = Line2D([0], [0], marker='s', color='w', label='Obstacles',
                            markerfacecolor='grey', markersize=10, alpha=0.3)
    proxy_path = Line2D([0], [0], color='blue', lw=3, label='Path')
    proxy_start = Line2D([0], [0], marker='o', color='w', label='Start',
                         markerfacecolor='green', markersize=10)
    proxy_goal = Line2D([0], [0], marker='x', color='purple', label='Goal',
                        markersize=10, linestyle='None')

    # 添加图例
    #ax.legend(handles=[proxy_obstacle, proxy_path, proxy_start, proxy_goal])

    # 设置视角
    ax.view_init(elev=elev, azim=azim)

    plt.show()

# ----------------------- 定义障碍物类 -----------------------

class Obstacle:
    def __init__(self, lat, lon, alt, trunk_radius, trunk_height, obstacle_type='tree'):
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.trunk_radius = trunk_radius
        self.trunk_height = trunk_height
        self.type = obstacle_type  # 'tree', 'cube', 'sphere' etc.

    def add_to_grid(self, grid, home_easting, home_northing, transformer, offset_x, offset_y):
        local_pos = convert_geo_to_local(
            self.lat,
            self.lon,
            self.alt,
            home_easting,
            home_northing,
            transformer,
            offset_x,
            offset_y,
            grid.shape
        )
        if self.type == 'tree':
            add_tree_obstacle(grid, local_pos, self.trunk_radius, self.trunk_height)
            print(f"添加树形障碍物：中心={local_pos}, 树干半径={self.trunk_radius}米, 树干高度={self.trunk_height}米")
        elif self.type == 'cube':
            add_cube_obstacle(grid, local_pos, self.trunk_radius, self.trunk_height, obstacle_code=3)
            print(f"添加立方体障碍物：中心={local_pos}, 尺寸={self.trunk_radius}米, 高度={self.trunk_height}米")
        elif self.type == 'sphere':
            add_sphere_obstacle(grid, local_pos, self.trunk_radius, self.trunk_height, obstacle_code=4)
            print(f"添加球形障碍物：中心={local_pos}, 半径={self.trunk_radius}米, 高度={self.trunk_height}米")
        else:
            print(f"未知障碍物类型: {self.type}")

def add_cylinder_obstacle(grid, center, radius, height, obstacle_code=1):
    """
    在网格中添加圆柱形障碍物。

    :param grid: 三维网格
    :param center: 圆柱中心坐标 (x, y, z)
    :param radius: 圆柱半径（米）
    :param height: 圆柱高度（米）
    :param obstacle_code: 障碍物类型代码
    """
    x0, y0, z0 = center
    x0, y0, z0 = int(round(x0)), int(round(y0)), int(round(z0))
    for x in range(max(0, x0 - radius), min(grid.shape[0], x0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(grid.shape[1], y0 + radius + 1)):
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) <= radius:
                for z in range(max(0, z0), min(grid.shape[2], z0 + height)):
                    grid[x, y, z] = obstacle_code

def add_cube_obstacle(grid, center, size, height, obstacle_code=3):
    """
    在网格中添加立方体障碍物。

    :param grid: 三维网格
    :param center: 立方体中心坐标 (x, y, z)
    :param size: 立方体边长的一半（米）
    :param height: 立方体高度（米）
    :param obstacle_code: 障碍物类型代码
    """
    x0, y0, z0 = center
    x0, y0, z0 = int(round(x0)), int(round(y0)), int(round(z0))
    for x in range(max(0, x0 - size), min(grid.shape[0], x0 + size + 1)):
        for y in range(max(0, y0 - size), min(grid.shape[1], y0 + size + 1)):
            for z in range(max(0, z0), min(grid.shape[2], z0 + height)):
                grid[x, y, z] = obstacle_code

def add_sphere_obstacle(grid, center, radius, height, obstacle_code=4):
    """
    在网格中添加球形障碍物（仅考虑高度方向上的球体截面）。

    :param grid: 三维网格
    :param center: 球体中心坐标 (x, y, z)
    :param radius: 球体半径（米）
    :param height: 球体高度（米）
    :param obstacle_code: 障碍物类型代码
    """
    x0, y0, z0 = center
    x0, y0, z0 = int(round(x0)), int(round(y0)), int(round(z0))
    for x in range(max(0, x0 - radius), min(grid.shape[0], x0 + radius + 1)):
        for y in range(max(0, y0 - radius), min(grid.shape[1], y0 + radius + 1)):
            for z in range(max(0, z0 - radius), min(grid.shape[2], z0 + radius + 1)):
                if np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) <= radius:
                    grid[x, y, z] = obstacle_code

def add_tree_obstacle(grid, center, trunk_radius, trunk_height):
    """
    在网格中添加树形障碍物，包括树干和树冠。

    :param grid: 三维网格
    :param center: 树干中心坐标 (x, y, z)
    :param trunk_radius: 树干半径（米）
    :param trunk_height: 树干高度（米）
    """
    x0, y0, z0 = center
    x0, y0, z0 = int(round(x0)), int(round(y0)), int(round(z0))

    # 添加树干（圆柱体），障碍物代码为1
    for x in range(max(0, x0 - trunk_radius), min(grid.shape[0], x0 + trunk_radius + 1)):
        for y in range(max(0, y0 - trunk_radius), min(grid.shape[1], y0 + trunk_radius + 1)):
            if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) <= trunk_radius:
                for z in range(max(0, z0), min(grid.shape[2], z0 + trunk_height)):
                    grid[x, y, z] = 1  # 树干

    # 添加多层球形树冠，障碍物代码为2
    crown_layers = 3  # 树冠层数
    crown_radius = trunk_radius * 3  # 树冠半径
    for layer in range(crown_layers):
        layer_height = trunk_height + (layer * 5)  # 每层之间的高度间隔
        current_crown_z = z0 + trunk_height + layer * 5
        layer_radius = crown_radius - (layer * 1)  # 每层半径递减

        if layer_radius <= 0:
            break  # 防止半径为负

        for x in range(max(0, x0 - layer_radius), min(grid.shape[0], x0 + layer_radius + 1)):
            for y in range(max(0, y0 - layer_radius), min(grid.shape[1], y0 + layer_radius + 1)):
                for z in range(max(0, current_crown_z - 2), min(grid.shape[2], current_crown_z + 3)):
                    if np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - current_crown_z) ** 2) <= layer_radius:
                        grid[x, y, z] = 2  # 树冠

    # 添加分支（可选）
    add_branches(grid, center, trunk_radius, trunk_height)

def add_branches(grid, center, trunk_radius, trunk_height):
    """
    在树冠部分添加分支，以增加逼真度。

    :param grid: 三维网格
    :param center: 树干中心坐标 (x, y, z)
    :param trunk_radius: 树干半径（米）
    :param trunk_height: 树干高度（米）
    """
    x0, y0, z0 = center
    num_branches = 8  # 分支数量
    branch_length = 5  # 分支长度
    branch_radius = 1  # 分支半径

    for i in range(num_branches):
        angle = (2 * np.pi / num_branches) * i
        dx = int(round(np.cos(angle) * (trunk_radius + 1)))
        dy = int(round(np.sin(angle) * (trunk_radius + 1)))
        branch_start = (x0 + dx, y0 + dy, z0 + trunk_height)
        branch_end = (x0 + dx + int(round(np.cos(angle) * branch_length)),
                      y0 + dy + int(round(np.sin(angle) * branch_length)),
                      z0 + trunk_height + int(round(branch_length / 2)))

        # 使用 Bresenham3D 算法绘制分支
        branch_points = bresenham3D(branch_start, branch_end)
        for point in branch_points:
            if 0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1] and 0 <= point[2] < grid.shape[2]:
                grid[point[0], point[1], point[2]] = 1  # 分支也作为树干的一部分

# ----------------------- 路径平滑函数 -----------------------

def smooth_path(path: List[Tuple[int, int, int]], grid: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    使用线性插值或其他平滑方法对路径进行平滑处理。

    :param path: 原始路径列表
    :param grid: 网格数据
    :return: 平滑后的路径列表
    """
    if not path:
        return []

    smoothed_path = [path[0]]  # 保留起点

    index = 0
    while index < len(path) - 1:
        next_index = len(path) - 1  # 从后往前尝试最大跨度
        while next_index > index + 1:
            if is_straight_path(path[index], path[next_index], grid):
                break
            next_index -= 1
        smoothed_path.append(path[next_index])
        index = next_index

    return smoothed_path

def is_straight_path(start: Tuple[int, int, int], end: Tuple[int, int, int], grid: np.ndarray) -> bool:
    """
    检查从start到end的直线路径是否有障碍物。

    使用Bresenham算法的3D扩展来检测路径。

    :param start: 起点坐标
    :param end: 终点坐标
    :param grid: 网格数据
    :return: 如果路径无障碍，返回True，否则返回False
    """
    # 使用线性插值检查路径
    line = bresenham3D(start, end)
    for point in line:
        if grid[point[0], point[1], point[2]] != 0:
            return False
    return True

def bresenham3D(start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    3D Bresenham算法生成两点之间的直线路径。

    :param start: 起点坐标
    :param end: 终点坐标
    :return: 路径点列表
    """
    x1, y1, z1 = start
    x2, y2, z2 = end
    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        py = 2 * dy - dx
        pz = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if py >= 0:
                y1 += ys
                py -= 2 * dx
            if pz >= 0:
                z1 += zs
                pz -= 2 * dx
            py += 2 * dy
            pz += 2 * dz
            points.append((x1, y1, z1))

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        px = 2 * dx - dy
        pz = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if px >= 0:
                x1 += xs
                px -= 2 * dy
            if pz >= 0:
                z1 += zs
                pz -= 2 * dy
            px += 2 * dx
            pz += 2 * dz
            points.append((x1, y1, z1))

    # Driving axis is Z-axis
    else:
        px = 2 * dx - dz
        py = 2 * dy - dz
        while z1 != z2:
            z1 += zs
            if px >= 0:
                x1 += xs
                px -= 2 * dz
            if py >= 0:
                y1 += ys
                py -= 2 * dz
            px += 2 * dx
            py += 2 * dy
            points.append((x1, y1, z1))

    return points

# ----------------------- 主函数 -----------------------

def main():
    # ------------------- 配置参数 -------------------
    # 定义Home位置的地理坐标（起点）
    home_geo = {
        'lat': 31 + 17/60 + 10.3439/3600,  # 31°17'10.3439"N
        'lon': 121 + 30/60 + 38.2389/3600,  # 121°30'38.2389"E
        'alt': 0  # 起始高度（米）
    }

    # 创建Transformer对象，将地理坐标转换为本地UTM坐标
    try:
        utm_zone = int((home_geo['lon'] + 180) / 6) + 1
        is_northern = home_geo['lat'] >= 0
        if is_northern:
            epsg_code = 32600 + utm_zone  # 例如，UTM Zone 51N -> 32651
        else:
            epsg_code = 32700 + utm_zone  # 例如，UTM Zone 51S -> 32751
        utm_crs = f"EPSG:{epsg_code}"
        geographic_crs = "EPSG:4326"  # WGS84

        transformer_to_utm = Transformer.from_crs(geographic_crs, utm_crs, always_xy=True)
        transformer_to_geo = Transformer.from_crs(utm_crs, geographic_crs, always_xy=True)

        home_easting, home_northing = transformer_to_utm.transform(home_geo['lon'], home_geo['lat'])
        print(f"Home位置UTM坐标: Easting={home_easting}, Northing={home_northing}")

        # **测试转换回地理坐标**
        test_lat, test_lon = transformer_to_geo.transform(home_easting, home_northing)
        print(f"测试转换Home坐标: lat={test_lat}, lon={test_lon}")
    except Exception as e:
        print(f"Transformer创建错误: {e}")
        sys.exit(1)

    # ------------------- 定义网格大小和分辨率 -------------------
    # 根据实际场景调整网格大小和分辨率
    grid_size_x = 200  # 米
    grid_size_y = 200  # 米
    grid_size_z = 100  # 米
    grid_resolution = 1  # 每个网格单元代表1米

    # 计算网格形状
    grid_shape = (int(grid_size_x / grid_resolution),
                  int(grid_size_y / grid_resolution),
                  int(grid_size_z / grid_resolution))
    grid = np.zeros(grid_shape, dtype=int)

    # 定义坐标偏移量以处理负坐标
    # 例如，将坐标范围定义为x: -100到+100, y: -100到+100
    offset_x = grid_shape[0] // 2  # 100
    offset_y = grid_shape[1] // 2  # 100

    # ------------------- 定义障碍物列表 -------------------
    # 开发者只需在此列表中添加或修改障碍物配置
    obstacles = [
        Obstacle(
            lat=31 + 17/60 + 9.9474/3600,  # 31°17'9.9474"N
            lon=121 + 30/60 + 38.0247/3600,  # 121°30'38.0247"E
            alt=0,
            trunk_radius=3,
            trunk_height=15,
            obstacle_type='tree'
        ),
        Obstacle(
            lat=31 + 17/60 + 9.6815/3600,  # 31°17'9.6815"N
            lon=121 + 30/60 + 37.9426/3600,  # 121°30'37.9426"E
            alt=0,
            trunk_radius=3,
            trunk_height=15,
            obstacle_type='tree'
        ),
        # 示例：添加一个立方体障碍物
        Obstacle(
            lat=31 + 17/60 + 9.5/3600,  # 31°17'9.5"N
            lon=121 + 30/60 + 38.5/3600,  # 121°30'38.5"E
            alt=0,
            trunk_radius=5,  # 这里的 trunk_radius 表示半边长
            trunk_height=10,
            obstacle_type='cube'
        ),
        # 示例：添加一个球形障碍物
        Obstacle(
            lat=31 + 17/60 + 9.3/3600,  # 31°17'9.3"N
            lon=121 + 30/60 + 38.7/3600,  # 121°30'38.7"E
            alt=0,
            trunk_radius=5,
            trunk_height=10,
            obstacle_type='sphere'
        ),
    ]

    # ------------------- 添加障碍物到网格 -------------------
    for obstacle in obstacles:
        obstacle.add_to_grid(grid, home_easting, home_northing, transformer_to_utm, offset_x, offset_y)

    # ------------------- 定义必须经过的途径点 -------------------
    must_pass_geo = [
        {
            'lat': 31 + 17/60 + 9.9474/3600,  # 31°17'9.9474"N
            'lon': 121 + 30/60 + 38.0247/3600,  # 121°30'38.0247"E
            'alt': 30  # 高度30米
        },
        {
            'lat': 31 + 17/60 + 9.6815/3600,  # 31°17'9.6815"N
            'lon': 121 + 30/60 + 37.9426/3600,  # 121°30'37.9426"E
            'alt': 30  # 高度30米
        }
    ]

    # ------------------- 定义起点和终点 -------------------
    start_geo = {
        'lat': 31 + 17/60 + 10.3439/3600,  # 31°17'10.3439"N
        'lon': 121 + 30/60 + 38.2389/3600,  # 121°30'38.2389"E
        'alt': 0  # 飞行高度（米）
    }
    goal_geo = {
        'lat': 31 + 17/60 + 8.1979/3600,  # 31°17'8.1979"N
        'lon': 121 + 30/60 + 37.1847/3600,  # 121°30'37.1847"E
        'alt': 0  # 飞行高度（米）
    }

    # ------------------- 定义所有途径点（包括必须经过的途径点） -------------------
    waypoints_geo = must_pass_geo  # 途径点列表

    # ------------------- 转换所有途径点为本地坐标 -------------------
    waypoints_local = []
    for wp in waypoints_geo:
        local_wp = convert_geo_to_local(
            wp['lat'],
            wp['lon'],
            wp['alt'],
            home_easting,
            home_northing,
            transformer_to_utm,
            offset_x,
            offset_y,
            grid_shape
        )
        waypoints_local.append(local_wp)
        print(f"必须经过的途径点本地坐标：{local_wp}")

    # ------------------- 转换起点和终点为本地坐标 -------------------
    start = convert_geo_to_local(
        start_geo['lat'],
        start_geo['lon'],
        start_geo['alt'],
        home_easting,
        home_northing,
        transformer_to_utm,
        offset_x,
        offset_y,
        grid_shape
    )
    goal = convert_geo_to_local(
        goal_geo['lat'],
        goal_geo['lon'],
        goal_geo['alt'],
        home_easting,
        home_northing,
        transformer_to_utm,
        offset_x,
        offset_y,
        grid_shape
    )

    print(f"起点本地坐标：{start}")
    print(f"终点本地坐标：{goal}")

    # ------------------- 检查起点和终点是否在网格内 -------------------
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1] and 0 <= start[2] < grid.shape[2]):
        print("起点超出网格范围。")
        sys.exit(1)
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1] and 0 <= goal[2] < grid.shape[2]):
        print("终点超出网格范围。")
        sys.exit(1)

    # ------------------- 检查起点和终点是否为障碍物 -------------------
    if grid[start[0]][start[1]][start[2]] != 0:
        print("起点位于障碍物上，请重新选择。")
        sys.exit(1)
    if grid[goal[0]][goal[1]][goal[2]] != 0:
        print("终点位于障碍物上，请重新选择。")
        sys.exit(1)

    # ------------------- 检查途径点是否为障碍物 -------------------
    for idx, wp in enumerate(waypoints_local):
        if grid[wp[0]][wp[1]][wp[2]] != 0:
            print(f"途径点 {idx} 位于障碍物上，请重新选择。")
            sys.exit(1)

    # ------------------- 路径规划 -------------------
    print("\n开始运行A*算法进行路径规划...")
    full_path = []

    # 路径分段：起点 -> 途径点1 -> 途径点2 -> ... -> 终点
    segments = [start] + waypoints_local + [goal]
    for i in range(len(segments) - 1):
        segment_start = segments[i]
        segment_goal = segments[i + 1]
        print(f"\n规划路径段 {i+1}: 从 {segment_start} 到 {segment_goal} ...")
        path = a_star_3d(segment_start, segment_goal, grid)
        if path:
            print(f"路径段 {i+1} 规划成功。路径长度: {len(path)}")
            if i > 0:
                # 移除前一个段的最后一点以避免重复
                path = path[1:]
            full_path.extend(path)
        else:
            print(f"路径段 {i+1} 未找到路径。")
            sys.exit(1)

    # ------------------- 路径平滑处理 -------------------
    print("\n对路径进行平滑处理...")
    # 分段平滑，确保必须经过的途径点不会被移除
    smoothed_full_path = []
    segment_indices = [0]
    for i in range(len(segments) - 1):
        segment_start_index = sum(len(a_star_3d(segments[j], segments[j+1], grid)) - 1 for j in range(i))
        segment_end_index = segment_start_index + len(a_star_3d(segments[i], segments[i+1], grid)) - 1
        segment_indices.append(segment_end_index)
    # 使用segment_indices将full_path分段
    for i in range(len(segment_indices) - 1):
        segment = full_path[segment_indices[i]:segment_indices[i+1]+1]
        smoothed_segment = smooth_path(segment, grid)
        if i > 0:
            # 移除前一个段的最后一点以避免重复
            smoothed_segment = smoothed_segment[1:]
        smoothed_full_path.extend(smoothed_segment)


    # ------------------- 路径转换为航点 -------------------
    print("\n转换完整路径为航点...")
    waypoints = convert_path_to_waypoints(smoothed_full_path, home_easting, home_northing, transformer_to_geo, offset_x, offset_y)

    # ------------------- 写入航点文件 -------------------
    output_file = 'mission.waypoints'
    print("生成航点文件...")
    write_waypoints_file(waypoints, output_file)

    # ------------------- 可视化 -------------------
    print("进行可视化...")
    # 你可以根据需要调整 elev 和 azim 参数
    visualize(grid, path=smoothed_full_path, start=start, goal=goal, elev=30, azim=45)

    print("\n所有操作完成。")

if __name__ == "__main__":
    main()