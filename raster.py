"""
Растеризация треугольников с z-буфером.

Режимы:
  flat    — один цвет на весь треугольник (цвет по нормали грани);
  gouraud — интенсивности в вершинах, линейная интерполяция по пикселям.
"""

from __future__ import annotations

import math
from typing import Tuple

RGB = Tuple[int, int, int]
# Вершина экрана: (sx, sy, depth, intensity 0..1)
ScreenV = Tuple[float, float, float, float]


def hex_rgb(color: str) -> RGB:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


class FrameBuffer:
    """RGB + z-буфер. Вывод через PPM → PhotoImage (как в ЛР1/ЛР2)."""

    def __init__(self, width: int, height: int, bg: str = "#1a1a1a") -> None:
        self.width = max(1, width)
        self.height = max(1, height)
        self.bg = hex_rgb(bg)
        self.color = bytearray(self.width * self.height * 3)
        self.zbuf = [float("inf")] * (self.width * self.height)
        self.clear()

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, width), max(1, height)
        if width == self.width and height == self.height:
            return
        self.width, self.height = width, height
        self.color = bytearray(width * height * 3)
        self.zbuf = [float("inf")] * (width * height)
        self.clear()

    def clear(self) -> None:
        r, g, b = self.bg
        self.color[:] = bytes((r, g, b)) * (self.width * self.height)
        for i in range(len(self.zbuf)):
            self.zbuf[i] = float("inf")

    def put(self, x: int, y: int, z: float, rgb: RGB) -> None:
        """Пиксель с тестом глубины (меньше z — ближе)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        i = y * self.width + x
        if z < self.zbuf[i]:
            self.zbuf[i] = z
            j = i * 3
            self.color[j : j + 3] = bytes(rgb)

    def to_photo_data(self) -> bytes:
        header = f"P6 {self.width} {self.height} 255\n".encode("ascii")
        return header + self.color


def _interp_edge(
    y0: float, x0: float, z0: float, i0: float,
    y1: float, x1: float, z1: float, i1: float,
    y: int,
) -> Tuple[float, float, float]:
    """Интерполяция x, z, intensity на горизонтали y между двумя вершинами."""
    if abs(y1 - y0) < 1e-9:
        return x0, z0, i0
    t = (y - y0) / (y1 - y0)
    return (
        x0 + (x1 - x0) * t,
        z0 + (z1 - z0) * t,
        i0 + (i1 - i0) * t,
    )


def fill_triangle_gouraud(fb: FrameBuffer, v0: ScreenV, v1: ScreenV, v2: ScreenV) -> None:
    """
    Заливка треугольника с интерполяцией интенсивности (Гуро) и z.

    Сортируем вершины по Y, рисуем верхнюю и нижнюю половины scanline'ами.
    """
    pts = [v0, v1, v2]
    pts.sort(key=lambda p: p[1])  # по sy
    (x0, y0, z0, i0), (x1, y1, z1, i1), (x2, y2, z2, i2) = pts

    y_start = max(0, int(math.ceil(y0)))
    y_end = min(fb.height - 1, int(math.floor(y2)))

    for y in range(y_start, y_end + 1):
        # длинное ребро 0→2
        xa, za, ia = _interp_edge(y0, x0, z0, i0, y2, x2, z2, i2, y)
        if y < y1 or abs(y2 - y1) < 1e-9:
            # верхняя половина: ребро 0→1
            xb, zb, ib = _interp_edge(y0, x0, z0, i0, y1, x1, z1, i1, y)
        else:
            # нижняя: ребро 1→2
            xb, zb, ib = _interp_edge(y1, x1, z1, i1, y2, x2, z2, i2, y)

        if xa > xb:
            xa, xb = xb, xa
            za, zb = zb, za
            ia, ib = ib, ia

        x_left = max(0, int(math.ceil(xa)))
        x_right = min(fb.width - 1, int(math.floor(xb)))
        if x_right < x_left:
            continue
        span = xb - xa
        for x in range(x_left, x_right + 1):
            t = 0.0 if abs(span) < 1e-9 else (x - xa) / span
            z = za + (zb - za) * t
            inten = ia + (ib - ia) * t
            g = int(round(255.0 * (0.0 if inten < 0.0 else (1.0 if inten > 1.0 else inten))))
            fb.put(x, y, z, (g, g, g))


def fill_triangle_flat(
    fb: FrameBuffer,
    v0: Tuple[float, float, float],
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float],
    intensity: float,
) -> None:
    """Плоская закраска: одна интенсивность на весь треугольник."""
    i = 0.0 if intensity < 0.0 else (1.0 if intensity > 1.0 else intensity)
    fill_triangle_gouraud(
        fb,
        (v0[0], v0[1], v0[2], i),
        (v1[0], v1[1], v1[2], i),
        (v2[0], v2[1], v2[2], i),
    )
