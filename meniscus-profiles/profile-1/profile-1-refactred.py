import math
import numpy as np
from nptyping import NDArray, Shape, Float64
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TABLEAU_COLORS 
from typing import Tuple, Optional
from pathlib import Path

class BaseObject():
    def __init__(self):
        pass
    
    def length(self) -> float:
        """
            Длина геом.объекта. На основе нее вычисляется шаг сетки по количеству точек в сетке
        """
        raise NotImplementedError

    def plot_plt(ax: Axes) -> None:
        raise NotImplementedError

class Point():
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def culc_dist(self, point: "Point") -> float:
        p = point.get_coords()
        return math.sqrt((p[0]-self.x) ** 2 + (p[1]-self.y) ** 2)

    def get_coords(self) -> Tuple[float, float]:
        return self.x, self.y

    def plot_plt(self, ax: Axes, **kwargs):
        ax.scatter(self.x, self.y, **kwargs)

    def __str__(self):
        return f"Point({self.x}; {self.y})"
    
    def __repr__(self):
        return self.__str__()

class Circle(BaseObject):
    def __init__(self, center: Point, r: float):
        self.c = center
        self.r = r
        assert r > 0, ValueError("r must be positive")
    
    def get_center(self) -> Point:
        return self.c

    def get_r(self) -> float:
        return self.r

    def plot_plt(self, plt, ax: Axes) -> None:
        circle1 = plt.Circle(self.c.get_coords(), self.r, fill=False, linewidth=2)
        ax.add_patch(circle1)
    
    def __str__(self):
        return f"Circle({self.c}; {self.r})"
    
    def __repr__(self):
        return self.__str__()

class Line(BaseObject):
    """
        прямая y=kx+b
    """
    def __init__(self, k: float, b: float):
        self.k = k
        self.b = b
    
    def culc_y(self, x: float):
        return self.k * x + self.b
    
    def plot_plt(self, ax: Axes, **kwargs):
        x = [-1.0, 1.0]
        y = [self.culc_y(x[0]), self.culc_y(x[1])]
        ax.plot(x, y, **kwargs)
    
    def __str__(self):
        return f"y = {self.k}*x + {self.b}"
    
    def __repr__(self):
        return self.__str__()
    
class LineSegment(BaseObject):
    def __init__(self, point_start: Point, point_end: Point):
        self.p1 = point_start
        self.p2 = point_end
    
    def plot_plt(self, ax: Axes, **kwargs):
        point1 = self.p1.get_coords()
        point2 = self.p2.get_coords()
        x = [point1[0], point2[0]]
        y = [point1[1], point2[1]]
        ax.plot(x, y, **kwargs)
    
    def get_points(self) -> Tuple[Point, Point]:
        return self.p1, self.p2
    
    def length(self) -> float:
        return self.p1.culc_dist(self.p2)

def all_is_not_None(args: list):
    return all([arg is not None for arg in args])

class Arc(BaseObject):
    def __init__(self,
        center: Optional[Point] = None,
        r: Optional[float] = None,
        phi_start: Optional[float] = None,
        phi_end: Optional[float] = None,
        point_start: Optional[Point] = None,
        point_end: Optional[Point] = None,
        circle: Optional[Circle] = None,
        tol: float = 1e-9
    ):
        if not all_is_not_None([center, r]) and circle is not None:
            center = circle.get_center()
            r = circle.get_r()
        args1 = [center, r, phi_start, phi_end]
        args2 = [center, r, point_start, point_end]

        if all_is_not_None(args1):
            self.c = center
            self.r = r
            self.phi1 = phi_start
            self.phi2 = phi_end
            self.point_start = None
            self.point_end = None
        elif all_is_not_None(args2): 
            self.c = center
            self.r = r
            self.point_start = point_start
            self.point_end = point_end
            assert abs(center.culc_dist(point_start) - r) < tol, "point_start не лежит на дуге окружности с центром center и радиусом r"
            assert abs(center.culc_dist(point_end) - r) < tol, "point_end не лежит на дуге окружности с центром center и радиусом r"
            self.phi1, self.phi2 = self.culc_phi1_phi2(self.c, self.point_start, self.point_end)
        else:
            raise ValueError("Arc должен быть задан с помощью (center, r, phi_start, phi_end), или (center, r, point_start, point_end), или (circle, point_start, point_end) или (circle, point_start, point_end)")
    
    def culc_phi_from_center(self, c: Point, p: Point):
        """
            Вычисляет угол (полярные координаты) точки p относительно центра c
        """
        p = np.array(p.get_coords())
        c = np.array(c.get_coords())
        dr = p - c
        return np.arctan2(dr[1], dr[0])

    # print(culc_phi_from_center((1.0, 1.0), (1.0, 0.0)))  # -pi/2
    # exit()

    def culc_phi1_phi2(self, c: Point, p1: Point, p2: Point):
        """
            Вычисляет углы точек p1, p2 дуги окружнсоти с центром c
        """
        phi1 = self.culc_phi_from_center(c, p1)
        phi2 = self.culc_phi_from_center(c, p2)
        if phi2 < phi1:
            phi2 += 2 * np.pi

        return phi1, phi2

    def plot_plt(self, ax, n_points:int=30, **kwargs):
        """
            Рисует дугу окружности по центру, радиусу и углам (в радианах).

            Параметры:
            ----------
            ax : matplotlib.axes.Axes
                Объект Axes, на котором рисуем дугу.
            n_points : int
                Количество точек для построения (по умолчанию 100).
            **kwargs :
                Дополнительные параметры для plt.plot (например, color, linewidth, linestyle).
        """
        theta = np.linspace(self.phi1, self.phi2, n_points)
        center = self.c.get_coords()
        x = center[0] + self.r * np.cos(theta)
        y = center[1] + self.r * np.sin(theta)
        ax.plot(x, y, **kwargs)
    
    def length(self):
        return self.r * (self.phi2 - self.phi1)


class Geometry():
    """
        Класс, осуществлящий взаимодействие объектов
    """
    def __init__(self):
        pass

    def distance(point1: Point, point2: Point) -> float:
        p1 = point1.get_coords()
        p2 = point2.get_coords()
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_tangent(self, circ: Circle, line: Line, tol:float = 1e-9) -> bool:
        c = circ.get_center().get_coords()
        r = circ.get_r()
        k = line.k
        b = line.b
        if ((k*c[0] - c[1] + b)**2 - r**2 * (k**2 + 1)) < tol:
            return True
        return False

    def common_tangents(self, circ1: Circle, circ2: Circle) -> list[Line]:
        """
            circ1, circ2 -- окружности
            Возвращает список касательных y=kx+b, каждая как Line
        """
        x1, y1 = circ1.get_center().get_coords()
        x2, y2 = circ2.get_center().get_coords()
        r1 = circ1.get_r()
        r2 = circ2.get_r()
        assert r1 > 0, ValueError("r1 must be positive")
        assert r2 > 0, ValueError("r2 must be positive")
        tangents = []

        dx = x2 - x1
        dy = y2 - y1
        dist_sq = dx*dx + dy*dy

        if dist_sq == 0:
            # Центры совпадают
            return tangents
        # print("\n-----common_tangent-----")
        for sign1 in [+1, -1]:
            for sign2 in [+1, -1]:
                r = r2 * sign2 - r1 * sign1
                # print("r1, r2, r:", r1, r2, r)

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
                # print("a,b,c:", a,b,c)
                k = np.roots([a, b, c])
                b = sign2 * r2 * np.sqrt(k**2 + 1) + y2 - k * x2
                # print("k:", k)
                # print("b:", b)
                # print(k, b)
                for k_i, b_i in zip(k, b):
                    l = Line(k_i, b_i)
                    if self.is_tangent(circ1, l) and self.is_tangent(circ2, l):
                        tangents.append(l)
                        # print(l)
                        # print()
                # tangents.append((k[1], b[1]))
        # print("-----common_tangent_end-----\n")
        return tangents

    def is_similar_halfplane(self, p1: Point, p2: Point, line: Line) -> bool:
        c1 = p1.get_coords()
        c2 = p2.get_coords()
        k = line.k
        b = line.b
        if (k*c1[0] - c1[1] + b) * (k*c2[0] - c2[1] + b) > 0:
            return True
        return False

    def external_common_tangents(self, circ1: Circle, circ2: Circle) -> list[Line]:
        """
            circ1, circ2 -- окружности
            Возвращает список внешних касательных y=kx+b, каждая как Line
        """
        tangents = []
        common_tang = self.common_tangents(circ1, circ2)
        # print(len(common_tang))
        for line in common_tang:
            # print(line)
            if self.is_similar_halfplane(c1, c2, line):
                tangents.append(line)
        return tangents

    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#:~:text=or%20equivalently
    # TODO: проверки можно переписать красивее
    def tangent_point(self, circ: Circle, line: Line, tol=1e-9) -> Point:
        """
            circ: окружность
            line: прямая
            tol: tolerance for floating comparison
            Returns: Point(x_t, y_t) — точка касания
            Raises ValueError если линия не является касательной к окружности.
        """
        x0, y0 = circ.get_center().get_coords()
        r = circ.get_r()
        k = line.k
        b = line.b

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
        if not self.is_tangent(circ, line, tol=tol):
            raise ValueError(f"Прямая не касательна: расстояние {lhs_dist:.6g} != r {r:.6g}")
        
        x_t = x0 - (k * s) / denom
        y_t = y0 + s / denom
        return Point(x_t, y_t)
G = Geometry()

class BaseMeniscusContour(BaseObject):
    """
        Базовый класс для разных контуров мениска
    """
    def __init__(self):
        self.objs: Optional[list[BaseObject]] = None
        super().__init__()

class MeniscusContour1(BaseMeniscusContour):
    def __init__(self, circ_left: Circle, circ_right: Circle):
        super().__init__()
        self.circ1 = circ_left
        self.circ2 = circ_right

        # creating contour
        tangents = G.external_common_tangents(circ_left, circ_right)
        assert len(tangents) == 2, f"Error. find {len(tangents)} external common tangents"

        point1_1 = G.tangent_point(circ_left, tangents[0])
        point2_1 = G.tangent_point(circ_right, tangents[0])
        point1_2 = G.tangent_point(circ_left, tangents[1])
        point2_2 = G.tangent_point(circ_right, tangents[1])

        line_bottom = LineSegment(point1_2, point2_2)
        arc_right = Arc(circle=circ_right, point_start=point2_2, point_end=point2_1)
        line_top = LineSegment(point2_1, point1_1)
        arc_left = Arc(circle=circ_left, point_start=point1_1, point_end=point1_2)
        
        self.objs: list[BaseObject] = [line_bottom, arc_right, line_top, arc_left]
        self.l = 0.0
        for obj in self.objs:
            self.l += obj.length()
    
    def plot_plt(self, ax: Axes, **kwargs):
        for obj in self.objs:
            obj.plot_plt(ax, **kwargs)

    def length(self) -> float:
        return self.l

# TODO: подумать над названием
# Этот клас нужен, чтобы сетка меника была наследником BaseGrid. 
# Можно обойтись без этого класса, если BaseGrid переписать под лист объектов.
# С другой стороны, класс BaseGrid довольно логичный и для сохранения его чистоты 
# нужно ввести этот класс для сетки мениска
class MeniscusGeom1(BaseObject):
    """
        Описывает весь мениск как совокупность контуров
    """
    def __init__(
        self,
        width: float,
        h: float,
        circle_left: Circle,
        circle_right: Circle
    ):
        """
            TODO: подумать о том, как описывать мениск
            width - ширина
            h - расстояние между контурами
            circle_left - левый круг, задающий внешний контур
            circle_right - правый круг, задающий внешний контур
        """
        assert width > h, ValueError("width must be greater than h")
        assert circle_left.get_center().get_coords()[0] < circle_right.get_center().get_coords()[0], ValueError("circle _left must be to the left of the circle_right")
        assert circle_left.get_r() > width, ValueError("radius of the circle_left must be greater than width")
        assert circle_right.get_r() > width, ValueError("radius of the circle_right must be greater than width")        

        self.width = width
        self.h = h
        self.circle_left_out = circle_left
        self.circle_right_out = circle_right

        center_left = self.circle_left_out.get_center()
        center_right = self.circle_right_out.get_center()
        r_left_out = self.circle_left_out.get_r()
        r_right_out = self.circle_right_out.get_r() 
        r_left_inner = r_left_out - self.width
        r_right_inner = r_right_out - self.width

        self.contours: list[MeniscusContour1] = []
        for dr in np.arange(0.0, self.width + self.h, self.h, dtype=np.float64):
            r_left = r_left_inner + dr
            r_right = r_right_inner + dr
            circ_left = Circle(center_left, r_left)
            circ_right = Circle(center_right, r_right)
            c = MeniscusContour1(circ_left, circ_right)
            self.contours.append(c)

        # culc length
        # длина среднего контура, чотбы сетка была "ровнее"
        r_left_middle = (r_left_inner + r_left_out) / 2
        r_right_middle = (r_right_inner + r_right_out) / 2
        circ_left_middle = Circle(center_left, r_left_middle)
        circ_right_middle = Circle(center_right, r_right_middle)
        c_middle = MeniscusContour1(circ_left_middle, circ_right_middle)
        self.length_contour_middle = c_middle.length()

    def plot_plt(self, ax: Axes, **kwargs):
        for obj in self.contours:
            obj.plot_plt(ax, **kwargs)

    def length(self) -> float:
        """
            Эффективная длина. Нужна для вычисления шага сетки при известном количестве узлов в контуре
        """
        return self.length_contour_middle

class BaseGrid():
    def __init__(
        self, 
        obj: BaseObject,
        size: Optional[int] = None,
        h: Optional[float] = None
    ) -> None:
        """
            size - количество узлов в сетке
            h - шаг сетки
            Должен быть задан хотя бы один из параметров size, h. Приоритетнее size
        """
        self.size: Optional[int] = None
        self.length: Optional[float] = None
        self.h: Optional[float] = None
        self.grid: Optional[NDArray[Shape["*, 2"], Float64]] = None
        self.is_made: bool = False

        self.obj: Optional[BaseObject] = obj
        self.length = self.obj.length()
        if size is not None:
            self.size = size
            self._culc_h()
        elif h is not None:
            self.h = h
            self._culc_size()
        else:
            raise ValueError("size or h must be defined")

    def _make(self):
        """
            Создает сетку и помещает ее в self.grid
        """
        raise NotImplementedError

    def get_grid(self) -> NDArray[Shape["*, 2"], Float64]:
        if not self.is_made:
            self._make()
        return self.grid

    def get_grid_xy(self) -> Tuple[NDArray[Shape["*"], Float64], NDArray[Shape["*"], Float64]]:
        if not self.is_made:
            self._make()
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        # print(x.shape, x[:3])
        return x, y
    
    def _culc_h(self):
        if all_is_not_None([self.size, self.length]):
            self.h = self.length / (self.size - 1)
        else:
            raise ValueError("size or/and length are not defined")
    
    def _culc_size(self):
        if all_is_not_None([self.h, self.length]):
            self.size = int(self.length / self.h) + 1
            self._culc_h()  # update h
        else:
            raise ValueError("h or/and length are not defined")
    
    def plot_plt(self, ax: Axes, **kwargs):
        if not self.is_made:
            self._make()
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        ax.plot(x, y, **kwargs)
    
    def save_bin(self, dir: Path, filename_prefix: str) -> Tuple[Path, Path]:
        """
            Сохранение сетки в бинарники
            filename_prefix - префикс имени файла, будет добавлена ось и расширение
            dir - директория сохранения файлов

            Возвращает пути сохраенных файлов
        """
        if dir.exists():
            if not dir.is_dir():
                ValueError(f"'{dir}' существует, но не является директорией")
        else:
            dir.mkdir(parents=True)
        filename_template = f"{filename_prefix}_{{}}.bin"
        filename_x = filename_template.format('x')
        filename_y = filename_template.format('y')
        path_x = dir / filename_x
        path_y = dir / filename_y
        x, y = self.get_grid_xy()
        x.astype('f').tofile(path_x)
        y.astype('f').tofile(path_y)
        return path_x, path_y
        

# словарь соответствия геом.объектов и их сеток
grid_map: dict[type[BaseObject], type[BaseGrid]] = {}

class GridLineSegment(BaseGrid):
    def __init__(
        self,
        line_segment: LineSegment,
        size: Optional[int] = None,
        h: Optional[float] = None
    ) -> None:
        super().__init__(line_segment, size, h)

    def _make(self):
        if self.is_made:
            return
        if not isinstance(self.obj, LineSegment):
            raise ValueError("self.obj must be LineSegment") 
        point1, point2 = self.obj.get_points()
        p1 = point1.get_coords()
        p2 = point2.get_coords()
        x = np.linspace(p1[0], p2[0], self.size, endpoint=True)
        x = x.reshape((x.shape[0], 1))
        # print(x)
        y = np.linspace(p1[1], p2[1], self.size, endpoint=True)
        y = y.reshape((y.shape[0], 1))
        self.grid = np.hstack((x, y))
        self.is_made = True
grid_map[LineSegment] = GridLineSegment

class GridArc(BaseGrid):
    """
        Строит сетку для дуги окружности arc.
    """
    def __init__(
        self,
        arc: Arc,
        size: Optional[int] = None,
        h: Optional[float] = None,
        include_start_point: bool = False,
        include_end_point: bool = False
    ):
        """
            include_start_point = True - включать первую точку в сетку, 
            include_end_point = True - включать последнюю точку в сетку
        """
        super().__init__(arc, size, h)
        self.include_start_point = include_start_point
        self.include_end_point = include_end_point
    
    def _make(self) -> None:
        if self.is_made:
            return
        if not isinstance(self.obj, Arc):
            raise ValueError("self.obj must be Arc") 
        phi1 = self.obj.phi1
        phi2 = self.obj.phi2
        c = self.obj.c.get_coords()
        r = self.obj.r

        phi = np.linspace(phi1, phi2, self.size, endpoint=True)
        if not self.include_start_point:
            phi = phi[1:]
        if not self.include_end_point:
            phi = phi[:-1]

        x = c[0] + r * np.cos(phi)
        x = x.reshape((x.shape[0], 1))
        y = c[1] + r * np.sin(phi)
        y = y.reshape((y.shape[0], 1))
        self.grid = np.hstack((x, y))
        self.is_made = True
grid_map[Arc] = GridArc

# TODO: можно добавить аргумент size, правда непонятно, как его использовать
class GridMeniscusContour(BaseGrid):
    def __init__(
        self,
        contour: BaseMeniscusContour,
        # size: Optional[int] = None,
        h: float
    ) -> None:
        super().__init__(contour, h=h)
    
    def _make(self) -> None:
        if self.is_made:
            return
        if not isinstance(self.obj, BaseMeniscusContour):
            raise ValueError("self.obj must be BaseMeniscusContour or his children class")
        x = []
        y = []
        for obj in self.obj.objs:
            Grid_obj = grid_map[type(obj)](obj, h=self.h)
            xy = Grid_obj.get_grid()
            x += list(xy[:, 0])
            y += list(xy[:, 1])

        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        self.grid = np.hstack((x, y))
        self.is_made = True
grid_map[MeniscusContour1] = GridMeniscusContour

class GridMeniscus1(BaseGrid):
    def __init__(
        self,
        width: float,
        h: float,
        circle_left: Circle,
        circle_right: Circle
    ):
        """
            Сетка мениска с контуром №1
        """
        obj: MeniscusGeom1 = MeniscusGeom1(width, h, circle_left, circle_right)
        super().__init__(obj, h=h)   

    def _make(self) -> None:
        if self.is_made:
            return
        if not isinstance(self.obj, MeniscusGeom1):
            raise ValueError("self.obj must be MeniscusGeom1")
        x = []
        y = []
        for contours in self.obj.contours:
            Grid_obj = grid_map[type(contours)](contours, h=self.h)
            xy = Grid_obj.get_grid()
            x += list(xy[:, 0])
            y += list(xy[:, 1])

        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        self.grid = np.hstack((x, y))
        self.is_made = True 

if __name__ == "__main__":
    width = 0.05
    h = 0.01
    # colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
    colors = list(TABLEAU_COLORS.values())
    # print(colors)

    r1_out = 0.1
    c1 = Point(0, r1_out)
    circ1 = Circle(c1, r1_out)

    r2_out = 0.4
    c2 = Point(1, r2_out)
    circ2 = Circle(c2, r2_out)

    meniscus_grid = GridMeniscus1(width, h, circ1, circ2)
    path_x, path_y = meniscus_grid.save_bin(Path("./grids"), "meniscus")

    # fig, ax = plt.subplots()
    # plot_params = {
    #     "marker": 'o',
    #     "color": colors[0]
    # }
    # meniscus_grid.plot_plt(ax, **plot_params)

    # plt.show()