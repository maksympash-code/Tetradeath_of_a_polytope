# examples/gui.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

import random

from cg3d.pipeline import tetrahedralize_convex
from cg3d.mesh import TetMesh
from cg3d.hull import ConvexHull3D

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # потрібен для 'projection="3d"'


def build_mesh_from_tets(pts, tets):
    """
    Зібрати TetMesh з готових тетраедрів (індекси у pts).
    """
    mesh = TetMesh(pts)
    # 1) додаємо всі тетри
    for (i0, i1, i2, i3) in tets:
        mesh.add_tet(i0, i1, i2, i3)

    # 2) зшиваємо сусідів через facemap
    for key, lst in mesh.facemap.items():
        alive = [(t, f) for (t, f) in lst if mesh.tets[t].alive]
        if len(alive) == 2:
            (t1, f1), (t2, f2) = alive
            mesh.link(t1, f1, t2, f2)

    return mesh


def generate_random_points(n: int):
    """
    Генерує n випадкових точок в одиничному кубі [0,1]^3 + вершини куба,
    щоб оболонка була нормальною (опуклий куб).
    """
    pts = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    for _ in range(n):
        x = random.random()
        y = random.random()
        z = random.random()
        pts.append((x, y, z))
    return pts


def parse_points_from_text(text: str):
    """
    Парсить точки з багаторядкового тексту.
    Кожен рядок: x y z або x, y, z.
    Повертає список (x,y,z) як float.
    """
    points = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # пропускаємо пусті строки і коментарі
        line = line.replace(",", " ")
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Рядок {lineno}: очікується 3 числа, отримано: {len(parts)}")
        try:
            x, y, z = map(float, parts)
        except ValueError:
            raise ValueError(f"Рядок {lineno}: не вдалось прочитати числа '{line}'")
        points.append((x, y, z))
    if len(points) < 4:
        raise ValueError("Потрібно щонайменше 4 точки для 3D тетраедралізації.")
    return points


class TetraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tetradeath of a Polytope")
        self.geometry("800x650")

        # сюди покладемо Figure/Canvas
        self.fig = None
        self.ax = None
        self.canvas = None

        self._build_widgets()

    def _build_widgets(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        # --- Режим вводу ---
        mode_frame = ttk.LabelFrame(main, text="Режим вводу точок")
        mode_frame.pack(fill="x", pady=5)

        self.input_mode = tk.StringVar(value="random")

        random_rb = ttk.Radiobutton(
            mode_frame,
            text="Випадкові точки всередині куба",
            variable=self.input_mode,
            value="random",
            command=self._update_mode_state,
        )
        random_rb.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        manual_rb = ttk.Radiobutton(
            mode_frame,
            text="Ручне введення точок",
            variable=self.input_mode,
            value="manual",
            command=self._update_mode_state,
        )
        manual_rb.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # --- Параметри для random-режиму ---
        input_frame = ttk.LabelFrame(main, text="Параметри (для випадкових точок)")
        input_frame.pack(fill="x", pady=5)

        ttk.Label(input_frame, text="Кількість випадкових внутрішніх точок:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.n_entry = ttk.Entry(input_frame, width=10)
        self.n_entry.insert(0, "20")
        self.n_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # --- Поле для ручного вводу ---
        manual_frame = ttk.LabelFrame(main, text="Ручне введення точок (одна точка - один рядок)")
        manual_frame.pack(fill="both", expand=True, pady=5)

        self.points_text = tk.Text(manual_frame, height=6, wrap="none")
        self.points_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Підказка
        self.points_text.insert(
            "1.0",
            "# Приклад:\n"
            "# 0 0 0\n"
            "# 1 0 0\n"
            "# 1 1 0\n"
            "# 0 1 0\n"
            "# 0 0 1\n"
        )

        # --- Кнопка запуску ---
        run_btn = ttk.Button(main, text="Запустити тетраедралізацію", command=self.run_pipeline)
        run_btn.pack(fill="x", pady=10)

        # --- Результати ---
        result_frame = ttk.LabelFrame(main, text="Результати")
        result_frame.pack(fill="x", pady=5)

        self.vertices_var = tk.StringVar(value="—")
        self.surface_var = tk.StringVar(value="—")
        self.tets_var = tk.StringVar(value="—")
        self.valid_var = tk.StringVar(value="—")

        ttk.Label(result_frame, text="Вершини:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(result_frame, textvariable=self.vertices_var).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(result_frame, text="Граней оболонки:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(result_frame, textvariable=self.surface_var).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(result_frame, text="Тетраедрів:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(result_frame, textvariable=self.tets_var).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(result_frame, text="Валідація:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(result_frame, textvariable=self.valid_var).grid(row=3, column=1, sticky="w", padx=5, pady=2)

        info_label = ttk.Label(
            main,
            text="Файли hull.off, boundary.off, volume.vtk\n"
                 "будуть записані в поточну директорію.",
            foreground="gray",
            justify="center",
        )
        info_label.pack(fill="x", pady=5)

        # --- Фрейм для 3D-графіка ---
        plot_frame = ttk.LabelFrame(main, text="3D візуалізація")
        plot_frame.pack(fill="both", expand=True, pady=5)

        self.fig = Figure(figsize=(4, 3))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._update_mode_state()

    def _update_mode_state(self):
        """
        Вмикаємо/вимикаємо поля залежно від режиму вводу.
        """
        mode = self.input_mode.get()
        if mode == "random":
            self.n_entry.configure(state="normal")
        else:  # manual
            self.n_entry.configure(state="disabled")

    def update_plot(self, pts, tets):
        """
        Перемалювати 3D-графік у вікні для поточної тетраедралізації.
        """
        self.ax.clear()

        if not tets:
            self.ax.set_title("Немає тетраедрів")
            self.canvas.draw()
            return

        # малюємо всі ребра тетраедрів
        for (i0, i1, i2, i3) in tets:
            vs = [pts[i0], pts[i1], pts[i2], pts[i3]]
            edges = [(0, 1), (0, 2), (0, 3),
                     (1, 2), (1, 3),
                     (2, 3)]
            for a, b in edges:
                pa, pb = vs[a], vs[b]
                self.ax.plot(
                    [pa.x, pb.x],
                    [pa.y, pb.y],
                    [pa.z, pb.z],
                    linewidth=0.5,
                )

        # однакові масштаби
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        zs = [p.z for p in pts]
        if xs and ys and zs:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            min_z, max_z = min(zs), max(zs)
            max_range = max(
                max_x - min_x,
                max_y - min_y,
                max_z - min_z,
            )
            mx = 0.5 * (min_x + max_x)
            my = 0.5 * (min_y + max_y)
            mz = 0.5 * (min_z + max_z)
            if max_range == 0:
                max_range = 1.0
            self.ax.set_xlim(mx - max_range / 2, mx + max_range / 2)
            self.ax.set_ylim(my - max_range / 2, my + max_range / 2)
            self.ax.set_zlim(mz - max_range / 2, mz + max_range / 2)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Tetrahedralization (edges)")

        self.canvas.draw()

    def run_pipeline(self):
        mode = self.input_mode.get()

        # --- Вибір джерела точок ---
        if mode == "random":
            try:
                n = int(self.n_entry.get())
                if n < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Помилка", "Кількість точок має бути невід’ємним цілим числом.")
                return
            points = generate_random_points(n)
        else:  # manual
            raw_text = self.points_text.get("1.0", "end").strip()
            if not raw_text:
                messagebox.showerror("Помилка", "Введіть хоча б одну точку у текстове поле.")
                return
            try:
                points = parse_points_from_text(raw_text)
            except ValueError as e:
                messagebox.showerror("Помилка парсингу точок", str(e))
                return

        backend = "scipy"  # поки що тільки SciPy Delaunay

        try:
            # 1) Опукла оболонка + Делоне-тетраедралізація
            pts, surface, tets = tetrahedralize_convex(points, backend=backend)

            # 2) Зібрати TetMesh
            mesh = build_mesh_from_tets(pts, tets)

            # 3) Валідація
            report = mesh.validate()

            # 4) Експорти:
            hull = ConvexHull3D(pts)
            with open("hull.off", "w", encoding="utf-8") as f:
                f.write(hull.to_off())

            mesh.write_boundary_off("boundary.off")
            mesh.write_vtk_unstructured("volume.vtk")

            # 5) Оновити 3D-графік у вікні
            self.update_plot(pts, tets)

        except Exception as e:
            messagebox.showerror("Помилка виконання", str(e))
            return

        # --- Оновлюємо поля ---
        self.vertices_var.set(str(len(pts)))
        self.surface_var.set(str(len(surface)))
        self.tets_var.set(str(len(tets)))

        if report["bad_orientation"] or report["bad_face_multiplicity"] or report["bad_neighbors"]:
            self.valid_var.set("Є проблеми (див. консоль)")
        else:
            self.valid_var.set("OK")

        print("VALIDATION:", report)
        messagebox.showinfo(
            "Готово",
            "Тетраедралізація завершена.\n"
            "Записано файли:\n"
            "  - hull.off\n"
            "  - boundary.off\n"
            "  - volume.vtk",
        )


if __name__ == "__main__":
    app = TetraApp()
    app.mainloop()
