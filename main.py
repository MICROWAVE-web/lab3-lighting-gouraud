#!/usr/bin/env python3
"""
Лабораторная работа 3.
Построение сцен с простейшей моделью освещённости. Закраска Гуро.

Задание:
  1) фигура вращения из произвольной кривой;
  2) триангуляция поверхности;
  3) цвет треугольника — модель освещённости, градации серого (flat);
  4) движение камеры между двумя точками;
  5) закраска Гуро — сглаженное изображение.

Без OpenGL / NumPy: tkinter + свой растеризатор с z-буфером.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import List, Optional, Tuple

from lighting import lambert_intensity
from math3d import (
    Vec3,
    look_at,
    mat_vec,
    project_camera,
    rotate_z,
    transform_points,
    v_dot,
    v_lerp,
    v_normalize,
    v_sub,
)
from raster import FrameBuffer, fill_triangle_flat, fill_triangle_gouraud
from revolution import build_surface_of_revolution, triangle_normal


class LightingApp:
    BG = "#121212"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ЛР3 — Освещение | Фигура вращения | Гуро")
        self.root.geometry("1000x700")
        self.root.minsize(720, 520)
        self.root.configure(bg=self.BG)

        self.width = 760
        self.height = 560

        self.status = tk.Label(
            root,
            text="Загрузка…",
            anchor="w",
            padx=10,
            pady=6,
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Menlo", 11),
        )
        self.status.pack(side=tk.TOP, fill=tk.X)

        self.help = tk.Label(
            root,
            text=(
                "Пробел — камера A↔B | G — Гуро/flat | R — вращение объекта | "
                "[ ] — сектора | Esc — выход"
            ),
            anchor="w",
            padx=10,
            pady=6,
            bg="#1e293b",
            fg="#cbd5e1",
            font=("TkDefaultFont", 11),
        )
        self.help.pack(side=tk.BOTTOM, fill=tk.X)

        body = tk.Frame(root, bg=self.BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.info = tk.Text(
            body,
            width=34,
            height=20,
            bg="#0f172a",
            fg="#e2e8f0",
            font=("Menlo", 10),
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        self.info.pack(side=tk.RIGHT, fill=tk.Y)
        self.info.configure(state=tk.DISABLED)

        self.frame = tk.Frame(body, bg="#334155", padx=2, pady=2)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fb = FrameBuffer(self.width, self.height, self.BG)
        self.view = tk.Label(self.frame, bg=self.BG, bd=0)
        self.view.pack(fill=tk.BOTH, expand=True)

        # --- меш ---
        self.sectors = 24
        self._rebuild_mesh()

        # --- свет ---
        self.light_pos: Vec3 = (120.0, -80.0, 160.0)

        # --- камера: движение между двумя точками ---
        self.cam_a: Vec3 = (160.0, -200.0, 90.0)
        self.cam_b: Vec3 = (-180.0, -160.0, 40.0)
        self.cam_target: Vec3 = (0.0, 0.0, 0.0)
        self.cam_t = 0.0          # 0 → A, 1 → B
        self.cam_dir = 1.0        # туда-обратно
        self.cam_moving = True
        self.focal = 380.0

        # --- режимы ---
        self.gouraud = True       # False = flat shading
        self.spin_object = False
        self.spin_angle = 0.0
        self.frame_id = 0

        self.frame.bind("<Configure>", self._on_configure)
        self.root.bind("<Key>", self._on_key)
        self.view.focus_set()

        self.root.update_idletasks()
        self._sync_size()
        self._draw()
        self.root.after(33, self._tick)
        self.root.lift()
        try:
            self.root.attributes("-topmost", True)
            self.root.after(400, lambda: self.root.attributes("-topmost", False))
        except tk.TclError:
            pass

    def _rebuild_mesh(self) -> None:
        self.vertices, self.triangles, self.normals = build_surface_of_revolution(
            sectors=self.sectors
        )

    def _sync_size(self) -> None:
        w = self.frame.winfo_width() - 8
        h = self.frame.winfo_height() - 8
        if w > 40 and h > 40:
            self.width, self.height = w, h

    def _on_configure(self, _event: tk.Event) -> None:
        self._sync_size()
        if self.fb.width != self.width or self.fb.height != self.height:
            self.fb.resize(self.width, self.height)

    def camera_eye(self) -> Vec3:
        """Текущая позиция камеры на отрезке A—B."""
        # сглаживание (smoothstep), чтобы движение не было рывком
        t = self.cam_t
        t_smooth = t * t * (3.0 - 2.0 * t)
        return v_lerp(self.cam_a, self.cam_b, t_smooth)

    def _draw(self) -> None:
        if self.fb.width != self.width or self.fb.height != self.height:
            self.fb.resize(self.width, self.height)
        self.fb.clear()

        # опциональное вращение объекта вокруг Z
        verts = self.vertices
        norms = self.normals
        if self.spin_object and abs(self.spin_angle) > 1e-9:
            R = rotate_z(self.spin_angle)
            verts = transform_points(R, self.vertices)
            norms = []
            for n in self.normals:
                tv = mat_vec(R, (n[0], n[1], n[2], 0.0))
                norms.append(v_normalize((tv[0], tv[1], tv[2])))

        eye = self.camera_eye()
        view = look_at(eye, self.cam_target, up=(0.0, 0.0, 1.0))

        # вершины → СК камеры → экран (x, y, depth)
        cam_pts = transform_points(view, verts)
        screen: List[Tuple[float, float, float]] = []
        for p in cam_pts:
            sx, sy, depth = project_camera(
                p, focal=self.focal, screen_w=self.width, screen_h=self.height
            )
            screen.append((sx, sy, depth))

        # интенсивности в вершинах (мир) — для Гуро
        vertex_I: List[float] = [
            lambert_intensity(norms[i], verts[i], light_pos=self.light_pos)
            for i in range(len(verts))
        ]

        drawn = 0
        for tri in self.triangles:
            i0, i1, i2 = tri
            # отсечение тыльных граней: n · to_eye > 0
            fn = triangle_normal(verts, tri)
            to_eye = v_normalize(v_sub(eye, verts[i0]))
            if v_dot(fn, to_eye) <= 0.0:
                continue

            s0, s1, s2 = screen[i0], screen[i1], screen[i2]
            # отбрасываем треугольники полностью «за» камерой
            if s0[2] < 0.1 and s1[2] < 0.1 and s2[2] < 0.1:
                continue

            if self.gouraud:
                fill_triangle_gouraud(
                    self.fb,
                    (s0[0], s0[1], s0[2], vertex_I[i0]),
                    (s1[0], s1[1], s1[2], vertex_I[i1]),
                    (s2[0], s2[1], s2[2], vertex_I[i2]),
                )
            else:
                # flat: цвет по нормали грани и центру
                cx = (verts[i0][0] + verts[i1][0] + verts[i2][0]) / 3.0
                cy = (verts[i0][1] + verts[i1][1] + verts[i2][1]) / 3.0
                cz = (verts[i0][2] + verts[i1][2] + verts[i2][2]) / 3.0
                I = lambert_intensity(fn, (cx, cy, cz), light_pos=self.light_pos)
                fill_triangle_flat(self.fb, s0, s1, s2, I)
            drawn += 1

        # показать кадр
        import tkinter as tk_mod
        img = tk_mod.PhotoImage(data=self.fb.to_photo_data())
        self.view.configure(image=img)
        self.view.image = img  # keep ref

        self.frame_id += 1
        mode = "Gouraud" if self.gouraud else "flat"
        self.status.configure(
            text=(
                f"frame {self.frame_id} | {mode} | sectors={self.sectors} | "
                f"tris={len(self.triangles)} drawn={drawn} | "
                f"cam_t={self.cam_t:.2f} | cam={'move' if self.cam_moving else 'stop'}"
            )
        )
        self._update_info(eye, drawn)

    def _update_info(self, eye: Vec3, drawn: int) -> None:
        lines = [
            "ЛР3 — ОСВЕЩЕНИЕ",
            "================",
            "Фигура вращения",
            f"  секторов: {self.sectors}",
            f"  вершин:   {len(self.vertices)}",
            f"  треугольн: {len(self.triangles)}",
            f"  нарисовано: {drawn}",
            "",
            "Модель света:",
            "  I = Ia·ka + Id·kd·max(0,n·L)",
            "  палитра: grayscale",
            "",
            f"Закраска: {'ГУРО' if self.gouraud else 'FLAT'}",
            "  G — переключить",
            "",
            "Камера A ↔ B:",
            f"  A = ({self.cam_a[0]:.0f},{self.cam_a[1]:.0f},{self.cam_a[2]:.0f})",
            f"  B = ({self.cam_b[0]:.0f},{self.cam_b[1]:.0f},{self.cam_b[2]:.0f})",
            f"  eye=({eye[0]:.0f},{eye[1]:.0f},{eye[2]:.0f})",
            f"  t={self.cam_t:.2f}",
            "  Пробел — старт/стоп",
            "",
            "R — вращение объекта",
            "[ ] — число секторов",
        ]
        self.info.configure(state=tk.NORMAL)
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, "\n".join(lines))
        self.info.configure(state=tk.DISABLED)

    def _tick(self) -> None:
        try:
            if not self.view.winfo_exists():
                return
            if self.cam_moving:
                self.cam_t += 0.008 * self.cam_dir
                if self.cam_t >= 1.0:
                    self.cam_t = 1.0
                    self.cam_dir = -1.0
                elif self.cam_t <= 0.0:
                    self.cam_t = 0.0
                    self.cam_dir = 1.0
            if self.spin_object:
                self.spin_angle += 0.03
            self._draw()
        except tk.TclError:
            return
        except Exception as exc:
            self.status.configure(text=f"Ошибка: {exc}")
            import traceback
            traceback.print_exc()
            return
        self.root.after(33, self._tick)

    def _on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "escape":
            self.root.destroy()
        elif key == "space":
            self.cam_moving = not self.cam_moving
        elif key == "g":
            self.gouraud = not self.gouraud
        elif key == "r":
            self.spin_object = not self.spin_object
        elif key == "bracketright":  # ]
            self.sectors = min(48, self.sectors + 2)
            self._rebuild_mesh()
        elif key == "bracketleft":  # [
            self.sectors = max(8, self.sectors - 2)
            self._rebuild_mesh()


def main() -> None:
    root = tk.Tk()
    LightingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
