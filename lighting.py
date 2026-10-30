"""
Простейшая модель освещённости (градации серого).

I = Ia·ka + Id·kd · max(0, n · L)

  Ia, Id — интенсивности фона и диффузного источника (0..1);
  ka, kd — коэффициенты материала;
  n      — единичная нормаль;
  L      — единичный вектор К источнику света.

Цвет — оттенок серого: gray = int(255 * clamp(I, 0, 1)).
"""

from __future__ import annotations

from typing import Tuple

from math3d import Vec3, v_dot, v_normalize, v_sub


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def lambert_intensity(
    normal: Vec3,
    point: Vec3,
    *,
    light_pos: Vec3,
    ambient: float = 0.18,
    diffuse: float = 0.82,
    ka: float = 1.0,
    kd: float = 1.0,
) -> float:
    """
    Интенсивность в точке point с нормалью normal.
    Источник — точечный в light_pos (вектор L = light − point).
    """
    n = v_normalize(normal)
    L = v_normalize(v_sub(light_pos, point))
    ndotl = max(0.0, v_dot(n, L))
    I = ambient * ka + diffuse * kd * ndotl
    return clamp01(I)


def intensity_to_gray(I: float) -> int:
    """0..1 → 0..255 (палитра градаций серого)."""
    return int(round(255.0 * clamp01(I)))


def gray_rgb(I: float) -> Tuple[int, int, int]:
    g = intensity_to_gray(I)
    return (g, g, g)
