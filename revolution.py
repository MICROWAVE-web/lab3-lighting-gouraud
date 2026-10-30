"""
Фигура вращения + триангуляция поверхности.

Профильная кривая задаётся в плоскости XZ как набор точек (r, z),
где r ≥ 0 — расстояние от оси Z. Вращение вокруг оси Z на 360°.

Сетка: N сегментов по профилю × M секторов по углу.
Каждая «ячейка» сетки → 2 треугольника.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from math3d import Vec3, v_cross, v_normalize, v_sub

# Треугольник = тройка индексов вершин
Tri = Tuple[int, int, int]


def default_profile() -> List[Tuple[float, float]]:
    """
    Произвольная кривая-профиль (ваза / кувшин) в плоскости (r, z).

    Точки идут снизу вверх. r — радиус, z — высота.
    """
    return [
        (18.0, -70.0),
        (42.0, -55.0),
        (55.0, -30.0),
        (48.0, -5.0),
        (32.0, 15.0),
        (28.0, 35.0),
        (40.0, 55.0),
        (36.0, 70.0),
    ]


def build_surface_of_revolution(
    profile: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    sectors: int = 24,
) -> Tuple[List[Vec3], List[Tri], List[Vec3]]:
    """
    Построить меш фигуры вращения.

    Возвращает:
      vertices  — 3D-точки;
      triangles — индексы треугольников (ориентация наружу);
      normals   — нормаль на каждую ВЕРШИНУ (усреднение по смежным треугольникам),
                  нужна для закраски Гуро.
    """
    if profile is None:
        profile = default_profile()
    if sectors < 3:
        raise ValueError("sectors >= 3")
    n_prof = len(profile)
    if n_prof < 2:
        raise ValueError("в профиле нужно ≥ 2 точек")

    # --- вершины: для каждой точки профиля — sectors копий по углу ---
    vertices: List[Vec3] = []
    for r, z in profile:
        for j in range(sectors):
            ang = 2.0 * math.pi * j / sectors
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            vertices.append((x, y, z))

    def vid(i: int, j: int) -> int:
        """Индекс вершины: i — вдоль профиля, j — сектор."""
        return i * sectors + (j % sectors)

    # --- триангуляция: две треугольные грани на каждую ячейку сетки ---
    triangles: List[Tri] = []
    for i in range(n_prof - 1):
        for j in range(sectors):
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j)
            d = vid(i + 1, j + 1)
            # обход так, чтобы нормаль смотрела НАРУЖУ
            # (для r>0: против часовой при взгляде снаружи)
            triangles.append((a, c, b))
            triangles.append((b, c, d))

    # --- нормали вершин (для Гуро): сумма нормалей смежных треугольников ---
    normals = compute_vertex_normals(vertices, triangles)
    return vertices, triangles, normals


def compute_vertex_normals(
    vertices: Sequence[Vec3],
    triangles: Sequence[Tri],
) -> List[Vec3]:
    """Усреднённые нормали вершин по площадным нормалям треугольников."""
    acc = [(0.0, 0.0, 0.0) for _ in vertices]
    for i0, i1, i2 in triangles:
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        # нормаль ∝ площади (не нормализуем до суммирования — вес по площади)
        fn = v_cross(v_sub(p1, p0), v_sub(p2, p0))
        for idx in (i0, i1, i2):
            ax, ay, az = acc[idx]
            acc[idx] = (ax + fn[0], ay + fn[1], az + fn[2])
    return [v_normalize(n) for n in acc]


def triangle_normal(vertices: Sequence[Vec3], tri: Tri) -> Vec3:
    """Единичная нормаль одного треугольника."""
    i0, i1, i2 = tri
    return v_normalize(
        v_cross(v_sub(vertices[i1], vertices[i0]), v_sub(vertices[i2], vertices[i0]))
    )
