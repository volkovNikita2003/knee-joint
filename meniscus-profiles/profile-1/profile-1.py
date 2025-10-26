import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple


# line_bottom_coord = [(0.0, 0.0), (10.0, 0.0)]
# line_top

def common_tangents(c1, r1, c2, r2):
    """
    c1, c2: кортежи (x, y) центров окружностей
    r1, r2: радиусы окружностей
    Возвращает список касательных y=kx+b, каждая как (k, b)
    """
    x1, y1 = c1
    x2, y2 = c2
    tangents = []

    dx = x2 - x1
    dy = y2 - y1
    dist_sq = dx*dx + dy*dy

    if dist_sq == 0:
        # Центры совпадают
        return tangents

    for sign1 in [+1, -1]:
        for sign2 in [+1, -1]:
            r = r2 * sign2 - r1 * sign1
            if dist_sq - r*r < 0:
                # Не существует касательной
                continue
            # l = math.sqrt(dist_sq - r*r)
            # # Параметры направления касательной
            # vx = (dx * r + dy * l) / dist_sq
            # vy = (dy * r - dx * l) / dist_sq
            # # Точка касания на первой окружности
            # x_t1 = x1 + r1 * sign1 * vy
            # y_t1 = y1 - r1 * sign1 * vx
            # # Точка касания на второй окружности
            # x_t2 = x2 + r2 * sign2 * vy
            # y_t2 = y2 - r2 * sign2 * vx
            # tangents.append(((x_t1, y_t1), (x_t2, y_t2)))
            a = (dx**2 - r**2)
            b = -2 * dx * dy
            c = dy**2 - r**2
            k = np.roots([a, b, c])
            b = sign2 * r2 * np.sqrt(k**2 + 1) + y2 - k * x2
            # print(k, b)
            for k_i, b_i in zip(k, b):
                if is_tangent(c1, r1, k_i, b_i) and is_tangent(c2, r2, k_i, b_i):
                    tangents.append((k_i, b_i))
            # tangents.append((k[1], b[1]))
    return tangents

def is_tangent(c, r, k, b, tol=1e-9):
    if ((k*c[0] - c[1] + b)**2 - r**2 * (k**2 + 1)) < tol:
        return True
    return False

def is_similar_halfplane(c1, c2, line):
    k, b = line
    if (k*c1[0] - c1[1] + b) * (k*c2[0] - c2[1] + b) > 0:
        return True
    return False

def external_tangents(c1, r1, c2, r2):
    """
        c1, c2: кортежи (x, y) центров окружностей
        r1, r2: радиусы окружностей
        Возвращает список внешних касательных y=kx+b, каждая как (k, b)
    """
    tangents = []
    common_tang = common_tangents(c1, r1, c2, r2)
    for line in common_tang:
        if is_similar_halfplane(c1, c2, line):
            tangents.append(line)
    return tangents

# https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#:~:text=or%20equivalently
def tangent_point(center, r, k, b, tol=1e-9):
    """
    center: (x0, y0)
    r: radius (>=0)
    k, b: slope and intercept of line y = kx + b
         (if the line is vertical, pass k = None and b = x_line)
    tol: tolerance for floating comparison
    Returns: (x_t, y_t) — точка касания
    Raises ValueError если линия не является касательной к окружности.
    """

    x0, y0 = center
    if k is None:
        # vertical line x = b
        x_line = b
        dist = abs(x0 - x_line)
        if abs(dist - r) > tol:
            raise ValueError("Линия x = {:.6g} не является касательной (|dx|-r = {:.3g}).".format(x_line, dist-r))
        # точка касания: x = x_line, y = y0 (радиус горизонтален)
        return (x_line, y0)
    # non-vertical
    s = k*x0 - y0 + b
    denom = k*k + 1.0

    lhs_dist = abs(s) / math.sqrt(denom)
    if not is_tangent(center, r, k, b, tol=tol):
        raise ValueError(f"Прямая не касательна: расстояние {lhs_dist:.6g} != r {r:.6g}")
    
    x_t = x0 - (k * s) / denom
    y_t = y0 + s / denom
    return (x_t, y_t)

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_size(length: float, h: float) -> int:
    return int(length / h) + 1

def grid_line(p1: Tuple[float, float], p2: Tuple[float, float], size: int) -> list[list[float, float]]:
    x = np.linspace(p1[0], p2[0], size, endpoint=True)
    x = x.reshape((x.shape[0], 1))
    # print(x)
    y = np.linspace(p1[1], p2[1], size, endpoint=True)
    y = y.reshape((y.shape[0], 1))
    res = np.hstack((x, y))
    return res
# grid_line([0.0, 0.0], [1.0, 0.5], 5)
# exit()

def culc_phi_from_center(c: Tuple[float, float], p: Tuple[float, float]):
    """
        Вычисляет угол (полярные координаты) точки p относительно центра c
    """
    p = np.array(p)
    c = np.array(c)
    dr = p - c
    return np.arctan2(dr[1], dr[0])

# print(culc_phi_from_center((1.0, 1.0), (1.0, 0.0)))  # -pi/2
# exit()

def culc_phi1_phi2(c: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]):
    """
        Вычисляет углы точек p1, p2 дуги окружнсоти с центром c
    """
    phi1 = culc_phi_from_center(c, p1)
    phi2 = culc_phi_from_center(c, p2)
    if phi2 < phi1:
        phi2 += 2 * np.pi

    return phi1, phi2

def grid_arc(
    c: Tuple[float, float], 
    r: float, 
    phi1: float,
    phi2: float,
    size: int,
    start_point: bool = False,
    end_point: bool = False,
):
    """
        Строит сетку для дуги окружности с центром c радиуса r. 
        Дуга от phi1 радиан до phi2 радиан
        start_point = True - включать первую точку, 
        end_point = True - включать последнюю точку
    """
    phi = np.linspace(phi1, phi2, size, endpoint=True)
    if not start_point:
        phi = phi[1:]
    if not end_point:
        phi = phi[:-1]

    x = c[0] + r * np.cos(phi)
    x = x.reshape((x.shape[0], 1))
    y = c[1] + r * np.sin(phi)
    y = y.reshape((y.shape[0], 1))
    return np.hstack((x, y))
# res = grid_arc(0.0, 1.0, -np.pi / 2, 3 * np.pi / 4, 5)
# print(res)
# exit()


# TODO: передавать еще размеры каждого участка (отмечены break points)
def grid_contour(
    c1, 
    r1, 
    c2, 
    r2,
    size_bottom_line_long,
    size_arc2,
    size_top_line_long,
    size_arc1
):
    """
        Строит сетку для контура мениска
    """
    x = []
    y = []

    tangents = external_tangents(c1, r1, c2, r2)
    # tangents.sort(key=lambda x: )
    assert len(tangents) == 2, "error"
    
    point1_1 = tangent_point(c1, r1, tangents[0][0], tangents[0][1])
    point2_1 = tangent_point(c2, r2, tangents[0][0], tangents[0][1])
    point1_2 = tangent_point(c1, r1, tangents[1][0], tangents[1][1])
    point2_2 = tangent_point(c2, r2, tangents[1][0], tangents[1][1])
    # print(point1_1, point2_1)
    # print(point1_2, point2_2)

    # нижняя линия
    grid_bottom_line = grid_line(point1_2, point2_2, size_bottom_line_long)
    x += list(grid_bottom_line[:, 0])
    y += list(grid_bottom_line[:, 1])

    # арка окружности (c2, r2)
    phis = culc_phi1_phi2(c2, point2_2, point2_1)
    grid_arc2 = grid_arc(
        c2, 
        r2, 
        phis[0],
        phis[1],
        size_arc2
    )
    x += list(grid_arc2[:, 0])
    y += list(grid_arc2[:, 1])

    # верхняя линия
    grid_top_line = grid_line(point2_1, point1_1, size_top_line_long)
    x += list(grid_top_line[:, 0])
    y += list(grid_top_line[:, 1])

    # арка окружности (c1, r1)
    phis = culc_phi1_phi2(c1, point1_1, point1_2)
    grid_arc1 = grid_arc(
        c1,
        r1,
        phis[0],
        phis[1],
        size_arc1
    )
    x += list(grid_arc1[:, 0])
    y += list(grid_arc1[:, 1])

    # сдвиг сетки для периодических ГУ
    if size_bottom_line_long < 10:
        # надо подумать о том, какое минимальное требование нужно ставить. 
        # Это делается, чтобы сдвинуть начало сетки от стыка кривой части 
        # и сшивать в области прямой линии. 
        raise Exception("Слишком маленькая сетка")
    

    x = x[5:] + x[:5]
    x.append(x[0])
    y = y[5:] + y[:5]
    y.append(y[0])

    assert len(x) == len(y)
    x.reverse()
    y.reverse()
    return x, y

# Параметры
r1_out = 0.1
c1 = (0, r1_out)
r2_out = 0.4
c2 = (1, r2_out)
width = 0.05
h = 0.01
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']

# создание сетки
x = []
y = []
r1_inner = r1_out - width
r2_inner = r2_out - width
assert r1_inner > 0 and r2_inner > 0, "error"
h_radial = h
h_long = h
size_bottom_line = [0, 0]
size_top_line = [0, 0]
size_arc1 = [0, 0]
size_arc2 = [0, 0]

# Заданный шаг лучше всего сохраняется на внутреннем контуре. 
# В арках на внешних контурах шаг будет больше. 
tangents = external_tangents(c1, r1_inner, c2, r2_inner)
assert len(tangents) == 2, "error"

point1_1 = tangent_point(c1, r1_inner, tangents[0][0], tangents[0][1])
point2_1 = tangent_point(c2, r2_inner, tangents[0][0], tangents[0][1])
point1_2 = tangent_point(c1, r1_inner, tangents[1][0], tangents[1][1])
point2_2 = tangent_point(c2, r2_inner, tangents[1][0], tangents[1][1])

size_bottom_line[0] = get_size(distance(point1_2, point2_2), h_long)
phis2 = culc_phi1_phi2(c2, point2_2, point2_1)
size_arc2[0] = get_size(r2_inner * (phis2[1] - phis2[0]), h_radial)
size_top_line[0] = get_size(distance(point2_1, point1_1), h_long)
phis1 = culc_phi1_phi2(c1, point1_1, point1_2)
size_arc1[0] = get_size(r1_inner * (phis1[1] - phis1[0]), h_radial)

size_width = 0
size = [0, 0, 1]
for ind, dr in enumerate(np.arange(0.0, width + h, h, dtype=np.float64)):
    size[1] += 1
    r1 = r1_inner + dr
    r2 = r2_inner + dr
    
    x_contour, y_contour = grid_contour(
        c1, 
        r1, 
        c2, 
        r2,
        size_bottom_line_long = size_bottom_line[0],
        size_top_line_long=size_top_line[0],
        size_arc1=size_arc1[0],
        size_arc2=size_arc2[0]
    )
    x += x_contour
    y += y_contour
    if ind == 0:
        size[0] = len(x)
size_arc1[1] = size_width
size_arc2[1] = size_width
size_bottom_line[1] = size_width
size_top_line[1] = size_width
# добавление z для BinGridFactory
z = [0.0] * len(x)
print(len(x), len(x) / size[0])
print(f"size = {size[0]}, {size[1]}, {size[2]}")

# сохранение
np.array(x).astype('f').tofile('x_meniscus.bin')
np.array(y).astype('f').tofile('y_meniscus.bin')
np.array(z).astype('f').tofile('z_meniscus.bin')

# отрисовка
fig, ax = plt.subplots()
ax.plot(x, y, marker='o')
ax.set_aspect('equal')
plt.show()
exit()



# отрисовка
# Создаем фигуру и оси
fig, ax = plt.subplots()

for ind, dr in enumerate(np.arange(0.0, width + h, h, dtype=np.float64)):
    r1 = r1_out - dr
    r2 = r2_out - dr
    tangents = external_tangents(c1, r1, c2, r2)
    # Рисуем окружность
    circle1 = plt.Circle(c1, r1, color=colors[ind], fill=False, linewidth=2)
    ax.add_patch(circle1)

    circle2 = plt.Circle(c2, r2, color=colors[ind], fill=False, linewidth=2)
    ax.add_patch(circle2)

    # касательные и точки касания
    for i, (k, b) in enumerate(tangents, 1):
        # x = np.array([-1.0, 2.0])
        # y = k*x + b
        # ax.plot(x, y, marker='o')

        # точки касания
        point1 = tangent_point(c1, r1, k, b)
        point2 = tangent_point(c2, r2, k, b)
        # plt.scatter(point1[0], point1[1], color='red', s=20)
        # plt.scatter(point2[0], point2[1], color='red', s=20)
        x = [point1[0], point2[0]]
        y = [point1[1], point2[1]]
        ax.plot(x, y, marker='o', color=colors[ind])

ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-0.2, 1.5)
# ax.set_xlim(-2, 8)
# ax.set_ylim(-3, 5)-
ax.set_aspect('equal')

plt.show()