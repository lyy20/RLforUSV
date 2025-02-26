import heapq
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 定义节点类
class Node:
    def __init__(self, x, y, g=float('inf'), h=float('inf'), parent=None):
        self.x = x
        self.y = y
        self.g = g  # 从起点到当前节点的实际代价
        self.h = h  # 从当前节点到目标节点的估计代价
        self.f = g + h  # 总代价
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

# 计算启发式函数（欧几里得距离）
def heuristic(node, goal):
    return math.sqrt((node.x - goal[0]) ** 2 + (node.y - goal[1]) ** 2)

# 判断点是否在圆形障碍物内（考虑安全距离）
def is_in_obstacle(point, obstacles, safety_distance):
    x, y = point
    for obstacle in obstacles:
        obs_x, obs_y, radius = obstacle
        effective_radius = radius + safety_distance
        distance = math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
        if distance <= effective_radius:
            return True
    return False

# A*算法，连续空间搜索
def a_star_continuous(start, goal, obstacles, step_size=0.5, safety_distance=0.3):
    open_list = []
    closed_set = set()

    start_node = Node(start[0], start[1], g=0, h=heuristic(Node(start[0], start[1]), goal), parent=None)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if heuristic(current_node, goal) < step_size:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        # 考虑更多方向
        num_directions = 36
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            new_x = current_node.x + step_size * math.cos(angle)
            new_y = current_node.y + step_size * math.sin(angle)

            if not is_in_obstacle((new_x, new_y), obstacles, safety_distance):
                new_g = current_node.g + step_size
                new_node = Node(new_x, new_y, g=new_g, h=heuristic(Node(new_x, new_y), goal), parent=current_node)

                # 检查是否在开放列表中
                found = False
                for i, node in enumerate(open_list):
                    if math.sqrt((node.x - new_x) ** 2 + (node.y - new_y) ** 2) < step_size:
                        if new_g < node.g:
                            open_list[i] = new_node
                            heapq.heapify(open_list)
                        found = True
                        break

                if not found:
                    heapq.heappush(open_list, new_node)

    return None

# 计算当前需要朝向的角度
def calculate_direction_angle(current, next_point):
    dx = next_point[0] - current[0]
    dy = next_point[1] - current[1]
    angle = math.atan2(dy, dx) * 180 / math.pi
    return angle

# 可视化地图和路线
def visualize_map(obstacles, agent, target, path, safety_distance):
    plt.figure(figsize=(8, 8))

    # 绘制圆形障碍物及安全区域
    for obstacle in obstacles:
        x, y, radius = obstacle
        circle = Circle((x, y), radius, color='gray')
        plt.gca().add_patch(circle)
        safe_circle = Circle((x, y), radius + safety_distance, color='gray', alpha=0.2)
        plt.gca().add_patch(safe_circle)

    # 绘制智能体
    plt.scatter(agent[0], agent[1], color='blue', s=100, label='Agent')

    # 绘制目标
    plt.scatter(target[0], target[1], color='green', s=100, label='Target')

    # 绘制路径
    if path:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        plt.plot(path_x, path_y, color='red', linestyle='-', linewidth=2, label='Path')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Path Planning Visualization')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 主函数
def main():
    # 障碍物用 (圆心x, 圆心y, 半径) 表示
    obstacles = [(2, 2, 1), (5, 5, 1.5)]
    agent = (0, 0)  # 智能体坐标
    target = (9, 9)  # 目标坐标
    safety_distance = 0.3  # 安全距离

    # 运行A*算法
    path = a_star_continuous(agent, target, obstacles, safety_distance=safety_distance)

    if path and len(path) > 1:
        current_position = path[0]
        next_position = path[1]
        angle = calculate_direction_angle(current_position, next_position)
        print(f"智能体当前需要朝向的角度: {angle} 度")
    else:
        print("未找到路径或已到达目标")

    # 可视化地图和路线
    visualize_map(obstacles, agent, target, path, safety_distance)

if __name__ == "__main__":
    main()