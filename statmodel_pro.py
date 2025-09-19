# statmodel_pro.py — v3.41
# Новое в v3.41:
# - Правый график строится строго на [p_min, p_max] (без расширения).
# - Для ОТРИСОВКИ f(p) и p·f(p) используется PCHIP (shape-preserving cubic),
#   что устраняет видимые "изломы". Расчёты и симуляции остаются на линейной интерполяции.

from __future__ import annotations

import os, math
from typing import Optional, Dict
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.interpolate import PchipInterpolator   # <<< добавлено

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

__version__ = "3.41"

# ---------- утилиты (как в v3.40) ----------
def _knots_quantiles(x: np.ndarray, K: int) -> np.ndarray:
    x = np.asarray(x, float); K = max(4, int(K))
    qs = np.linspace(0, 1, K); qs[0], qs[-1] = 0.01, 0.99
    knots = np.quantile(x, qs); knots = np.unique(knots)
    if len(knots) < 4:
        mn, mx = float(np.min(x)), float(np.max(x))
        if not np.isfinite(mn) or not np.isfinite(mx): mn, mx = 0.0, 1.0
        if mx == mn: return np.array([0.0, 0.33, 0.66, 1.0], float)
        knots = np.linspace(mn, mx, 4)
    return knots

def rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    K = len(knots); assert K >= 4
    k1, kK_1, kK = knots[0], knots[-2], knots[-1]
    def d(a): a = np.asarray(a, float); return np.maximum(a, 0.0) ** 3
    cols = [x]
    for j in range(K - 2):
        kj = knots[j]
        cols.append(
            d(x - kj)
            - d(x - kK_1) * ((kK - kj) / (kK - kK_1))
            + d(x - kK)   * ((kK_1 - kj) / (kK - kK_1))
        )
    return np.column_stack(cols)

def hinge_basis(p01: np.ndarray, knots: np.ndarray) -> np.ndarray:
    p01 = np.asarray(p01, float); knots = np.asarray(knots, float)
    return np.maximum(p01[:, None] - knots[None, :], 0.0)

# ---------- загрузка Excel ----------
REQ_COLS = ["step", "start_day", "end_day", "base_price", "sales_total", "rev_total"]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sales")
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"В Excel отсутствуют требуемые колонки: {missing}. Ожидаются: {REQ_COLS}")
    df = df.copy()
    num_cols = ["step", "start_day", "end_day", "base_price", "sales_total", "rev_total"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].fillna(0)

    rem_cols = [c for c in df.columns if c.lower().startswith("remaining_k")]
    if rem_cols:
        drop_all_zero = df[rem_cols].eq(0).all(axis=1)
        df = df[~drop_all_zero].reset_index(drop=True)
    #print(f"[DEBUG] Rows after remaining_k* filter: {len(df)}")

    df["sales"]   = df["sales_total"].round().clip(lower=0).astype(int)
    df["revenue"] = df["rev_total"].astype(float).clip(lower=0)
    df["mid_day"] = 0.5 * (df["start_day"] + df["end_day"])
    df = df.sort_values(["step", "mid_day"], kind="mergesort").reset_index(drop=True)

    if not (df["base_price"].max() == 0 and df["base_price"].min() == 0):
        df["base_price"] = (df["base_price"].replace(0, np.nan).interpolate()
                            .bfill().ffill())
    return df

# ---------- GUI ----------
class StatModelGUI(tk.Tk):
    PAD_X = 8; PAD_Y = 8

    def __init__(self):
        super().__init__()
        self.title(f"Статмодель распродажи — v{__version__}")
        self.geometry("1400x900"); self.minsize(1100, 700)

        style = ttk.Style(self)
        for th in ("vista","xpnative","clam","default"):
            if th in style.theme_names(): style.theme_use(th); break
        style.configure("TButton", padding=(12,6))
        style.configure("TEntry", padding=(6,4))
        style.configure("Status.TLabel", anchor="w")
        style.configure("Header.TLabelframe.Label", font=("",10,"bold"))
        style.configure("Header.TLabel", font=("",10,"bold"))

        # Верхняя панель
        params = ttk.Labelframe(self, text=" Параметры ", style="Header.TLabelframe")
        params.pack(side=tk.TOP, fill=tk.X, padx=self.PAD_X, pady=(self.PAD_Y,0))

        ttk.Label(params, text="Файл:", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=(8,4), pady=6)
        self.path_var = tk.StringVar(value="results.xlsx")
        ttk.Entry(params, textvariable=self.path_var, width=22).grid(row=0, column=1, sticky="w", padx=4, pady=6)
        ttk.Button(params, text="Выбрать", command=self.on_choose_file).grid(row=0, column=2, padx=4, pady=6)
        self.btn_load = self._accent_button(params, "Загрузить", self.on_load_click)
        self.btn_load.grid(row=0, column=3, padx=(2,10), pady=6)

        ttk.Label(params, text="Дов. инт.:", style="Header.TLabel").grid(row=0, column=4, sticky="e", padx=(0,4), pady=6)
        self.ci_var = tk.StringVar(value="0.95")
        ttk.Entry(params, textvariable=self.ci_var, width=8).grid(row=0, column=5, sticky="w", padx=(0,12), pady=6)

        ttk.Label(params, text="Симуляций:", style="Header.TLabel").grid(row=0, column=6, sticky="e", padx=(0,4), pady=6)
        self.sim2_var = tk.StringVar(value="1000")
        ttk.Entry(params, textvariable=self.sim2_var, width=8).grid(row=0, column=7, sticky="w", padx=(0,12), pady=6)

        self.btn_family = ttk.Button(params, text="Аппрокс. спроса", command=self.on_approx_family)
        self.btn_family.grid(row=0, column=8, padx=(0,6), pady=6)

        self.btn_model = self._accent_button(params, "Моделирование", self.on_apply_model)
        self.btn_model.grid(row=0, column=9, padx=4, pady=6)
        self.btn_model.configure(state="disabled")

        for col in range(0, 10): params.columnconfigure(col, weight=0)
        params.columnconfigure(10, weight=1)

        # Основная панель
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=self.PAD_X, pady=self.PAD_Y)

        # Левая колонка (графики времени/кумулятивов)
        left_frame = ttk.Frame(paned); paned.add(left_frame, weight=3)
        self.fig_left = Figure(figsize=(9,8), dpi=100, constrained_layout=True)
        gs = self.fig_left.add_gridspec(3, 1, height_ratios=[1,1,1])
        self.ax_price = self.fig_left.add_subplot(gs[0,0])
        self.ax_sales = self.fig_left.add_subplot(gs[1,0], sharex=self.ax_price)
        self.ax_rev   = self.fig_left.add_subplot(gs[2,0], sharex=self.ax_price)
        self.canvas_left = FigureCanvasTkAgg(self.fig_left, master=left_frame)
        self.canvas_left.draw()
        self.canvas_left.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Правая колонка (f(p) и p·f(p))
        right_wrapper = ttk.Frame(paned); paned.add(right_wrapper, weight=2)
        self.fig_right = Figure(figsize=(6,7), dpi=100, constrained_layout=True)
        self.ax_demand = self.fig_right.add_subplot(111)
        self.ax_demand_r = None
        self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=right_wrapper)
        self.canvas_right.draw()
        self.canvas_right.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Статус-бар
        status = ttk.Frame(self)
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=self.PAD_X, pady=(0,self.PAD_Y))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=160)
        self.progress.pack(side=tk.RIGHT, padx=4)
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(status, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Состояния
        self.df: Optional[pd.DataFrame] = None
        self.selected_family = None
        self.family_cum_sales_line = None
        self.family_cum_rev_line   = None
        self.family_cum_sales_fill = None
        self.family_cum_rev_fill   = None
        self._cumfit_knots: Optional[np.ndarray] = None

        self.init_demand_axes()

    # ---- стилизованная кнопка
    def _accent_button(self, master, text, command) -> tk.Button:
        return tk.Button(master, text=text, command=command,
                         bg="#1f6feb", fg="white",
                         activebackground="#2563eb", activeforeground="white",
                         bd=0, relief="raised", font=("", 9, "bold"),
                         padx=12, pady=5, cursor="hand2")

    def set_busy(self, busy: bool, text: str = ""):
        if busy:
            self.configure(cursor="watch"); self.progress.start(10)
            if text: self.status_var.set(text)
        else:
            self.configure(cursor=""); self.progress.stop()
            if text: self.status_var.set(text)
        self.update_idletasks()

    def init_demand_axes(self):
        ax = self.ax_demand; ax.clear()
        if self.ax_demand_r is not None:
            try: self.ax_demand_r.remove()
            except Exception: pass
            self.ax_demand_r = None
        ax.set_title("Спрос и выручка как функции цены")
        ax.set_xlabel("Цена P"); ax.set_ylabel("Спрос за шаг f(p)")
        ax.grid(True, alpha=0.2)
        self.canvas_right.draw_idle()

    # ---- Загрузка
    def on_choose_file(self):
        init_path = self.path_var.get()
        init_dir = init_path if os.path.isdir(init_path) else (os.path.dirname(init_path) or os.getcwd())
        path = filedialog.askopenfilename(title="Выбрать Excel",
                                          filetypes=[("Excel файлы","*.xlsx"), ("Все файлы","*.*")],
                                          initialdir=init_dir)
        if not path: return
        self.path_var.set(path)
        self.status_var.set("Выбран файл. Нажмите «Загрузить».")

    def on_load_click(self):
        try:
            self.df = load_data(self.path_var.get())
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))
            self.status_var.set("Ошибка загрузки"); return
        self.selected_family = None
        self.btn_model.configure(state="disabled")
        self._remove_family_overlays()
        self.plot_data_graphs(); self.init_demand_axes()
        self.status_var.set("Данные загружены")

    def plot_data_graphs(self):
        if self.df is None: return
        df = self.df
        t = df["mid_day"].to_numpy(float)
        price = df["base_price"].to_numpy(float)
        cum_sales = df["sales"].cumsum().to_numpy()
        cum_rev   = df["revenue"].cumsum().to_numpy()

        ax = self.ax_price; ax.clear()
        ax.plot(t, price, marker="o", linestyle="-", label="Цена")
        ax.set_title("Цена от времени"); ax.set_xlabel("День (середина интервала)"); ax.set_ylabel("Цена")
        ax.grid(True, alpha=0.2); ax.legend()

        ax = self.ax_sales; ax.clear()
        ax.plot(t, cum_sales, marker="o", linestyle="-", label="Кумулятивные продажи (данные)")
        ax.set_title("Кумулятивные продажи"); ax.set_xlabel("День (середина интервала)"); ax.set_ylabel("Кум продажи")
        ax.grid(True, alpha=0.2); ax.legend()

        ax = self.ax_rev; ax.clear()
        ax.plot(t, cum_rev, linestyle="-", label="Кумулятивная выручка (данные)")
        ax.set_title("Кумулятивная выручка"); ax.set_xlabel("День (середина интервала)"); ax.set_ylabel("Кум. выручка")
        ax.grid(True, alpha=0.2); ax.legend()

        self.canvas_left.draw_idle()

    # ---- Кусочно-линейная аппроксимация f(p)
    def _families(self):
        def f_pl_cum(pp, *qk):
            x = np.asarray(pp, float)
            if self._cumfit_knots is None or len(self._cumfit_knots) != len(qk):
                return np.full_like(x, np.nan, float)
            qk = np.asarray(qk, float)
            return np.interp(x, self._cumfit_knots, qk, left=qk[0], right=qk[-1])
        return {"#PL*: Кусочно-линейная (кумфит)": dict(func=f_pl_cum)}

    def on_approx_family(self):
        if self.df is None:
            messagebox.showinfo("Нет данных", "Сначала нажмите «Загрузить»."); return

        res = self._fit_piecewise_cumfit()
        if res is None:
            messagebox.showwarning("Аппроксимация спроса", "Не удалось подобрать кусочно-линейную f(p)."); return

        f = self._families()["#PL*: Кусочно-линейная (кумфит)"]["func"]
        self.selected_family = dict(name="#PL*: Кусочно-линейная (кумфит)",
                                    func=f, p0=0.0, params=np.array(res["params"], float))
        self.btn_model.configure(state="normal")

        # ---- ПРАВЫЙ ГРАФИК: только на [p_min, p_max] + PCHIP для отображения
        ax = self.ax_demand; ax.clear()
        if self.ax_demand_r is not None:
            try: self.ax_demand_r.remove()
            except Exception: pass
            self.ax_demand_r = None

        price = self.df["base_price"].to_numpy(float)
        pmin, pmax = float(np.min(price)), float(np.max(price))
        xgrid = np.linspace(pmin, pmax, 600)  # без расширения диапазона

        qk = self.selected_family["params"]
        # базовая (линейная) — для подстраховки
        y_lin = f(xgrid, *qk)
        # PCHIP для красивой отрисовки (не используется в расчётах/симуляциях)
        try:
            pchip = PchipInterpolator(self._cumfit_knots, qk, extrapolate=False)
            y_plot = pchip(xgrid)
            y_plot = np.maximum(y_plot, 0.0)
        except Exception:
            y_plot = y_lin  # на случай, если что-то не так с узлами

        ax.plot(xgrid, y_plot, "-", linewidth=2.2, label="f(p)")
        ax.set_title("Спрос и выручка как функции цены")
        ax.set_xlabel("Цена P"); ax.set_ylabel("Спрос за шаг f(p)")
        ax.set_xlim(pmin, pmax); ax.grid(True, alpha=0.2)

        self.ax_demand_r = ax.twinx()
        y_pf = xgrid * y_plot
        self.ax_demand_r.plot(xgrid, y_pf, "--", linewidth=2.0, label="p·f(p)")
        self.ax_demand_r.set_ylabel("Выручка за шаг p·f(p)")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = self.ax_demand_r.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
        self.canvas_right.draw_idle()
        self.status_var.set("Аппроксимация спроса выполнена; правый график сглажён (PCHIP)")

    def _fit_piecewise_cumfit(self):
        if self.df is None: return None

        df = self.df
        price = df["base_price"].to_numpy(float)
        sales_obs = df["sales"].to_numpy(float)
        rev_obs   = df["revenue"].to_numpy(float)
        S_obs = np.cumsum(sales_obs); R_obs = np.cumsum(rev_obs)

        K = 12
        knots = np.quantile(price, np.linspace(0, 1, K)); knots = np.unique(knots)
        if len(knots) < 6: knots = np.linspace(price.min(), price.max(), 6)
        self._cumfit_knots = knots

        q0_scalar = max(1e-6, float(np.mean(sales_obs)))
        q0 = np.full(len(knots), q0_scalar, float)

        q_upper = max(1.5 * float(np.max(sales_obs)), q0_scalar*3.0, 10.0)
        bounds = [(0.0, q_upper)] * len(knots)

        cons = []
        for i in range(len(knots) - 1):
            cons.append({'type': 'ineq', 'fun': (lambda x, i=i: x[i] - x[i+1])})
        for i in range(len(knots) - 1):
            pi, pj = float(knots[i]), float(knots[i+1])
            cons.append({'type': 'ineq', 'fun': (lambda x, i=i, pi=pi, pj=pj: pi*x[i] - pj*x[i+1])})
        for i in range(len(knots) - 1):
            pi, pj = float(knots[i]), float(knots[i+1]); dp = max(1e-12, pj - pi)
            ai = 1.0 - pi/dp; bi = pi/dp
            cons.append({'type': 'ineq', 'fun': (lambda x, i=i, ai=ai, bi=bi: - (ai*x[i] + bi*x[i+1]))})

        S_last = max(S_obs[-1], 1e-6); R_last = max(R_obs[-1], 1e-6); lam = 1e-4
        def objective(qk: np.ndarray) -> float:
            q_t = np.interp(price, knots, qk, left=qk[0], right=qk[-1])
            S_hat = np.cumsum(q_t); R_hat = np.cumsum(q_t * price)
            eS = (S_hat - S_obs) / S_last; eR = (R_hat - R_obs) / R_last
            pen = lam * np.sum(np.diff(qk, n=2, prepend=qk[0], append=qk[-1])**2)
            return float(np.mean(eS*eS) + np.mean(eR*eR) + pen)

        res = minimize(objective, q0, method="SLSQP", bounds=bounds, constraints=cons,
                       options=dict(maxiter=10000, ftol=1e-12, disp=False))
        if not res.success: print("⚠️ Кумфит SLSQP:", res.message)
        return dict(params=np.clip(res.x, 0.0, q_upper))

    # ---- Моделирование по f(p) (как в v3.40) ----
    def on_apply_model(self):
        if self.selected_family is None or self.df is None:
            messagebox.showinfo("Нет аппроксимации", "Сначала выполните «Аппрокс. спроса»."); return

        try: ci = float(self.ci_var.get().replace(",", "."))
        except: ci = 0.95
        try: n_sims = int(float(self.sim2_var.get()))
        except: n_sims = 1000
        alpha = max(1e-6, 1.0 - float(ci))
        lo_pct, hi_pct = 100.0*(alpha/2.0), 100.0*(1.0 - alpha/2.0)

        df = self.df
        price = df["base_price"].to_numpy(float); T = len(price)
        t     = df["mid_day"].to_numpy(float)
        N0    = int(np.sum(df["sales"].to_numpy(int)))

        f   = self._families()["#PL*: Кусочно-линейная (кумфит)"]["func"]
        par = self.selected_family["params"]

        q_det = np.maximum(0.0, np.asarray(f(price, *par), float))
        s_det = np.zeros(T, float); R = float(N0)
        for i in range(T):
            take = min(q_det[i], R); s_det[i] = take; R -= take
            if R <= 0: s_det[i+1:] = 0.0; break
        cum_sales_det = np.cumsum(s_det)
        cum_rev_det   = np.cumsum(s_det * price)

        rng = np.random.default_rng(2027)
        sims_steps = np.zeros((n_sims, T), dtype=float)
        for r in range(n_sims):
            R = int(N0); s = np.zeros(T, float)
            for i in range(T):
                if R <= 0: break
                lam = float(q_det[i]); p = min(1.0, lam / max(R,1))
                draw = rng.binomial(R, p)
                s[i] = float(draw); R -= draw
            sims_steps[r] = s

        sims_cum     = np.cumsum(sims_steps, axis=1)
        sims_rev_cum = np.cumsum(sims_steps * price[None,:], axis=1)
        cum_sales_lo, cum_sales_hi = np.percentile(sims_cum,     [lo_pct, hi_pct], axis=0)
        cum_rev_lo,   cum_rev_hi   = np.percentile(sims_rev_cum, [lo_pct, hi_pct], axis=0)

        self._remove_family_overlays()

        self.family_cum_sales_line, = self.ax_sales.plot(
            t, cum_sales_det, linestyle="-.", linewidth=2.4,
            label="Моделирование: кумулятивные продажи"
        )
        self.family_cum_sales_fill = self.ax_sales.fill_between(
            t, cum_sales_lo, cum_sales_hi, alpha=0.18,
            label=f"Моделирование: ДИ {int(ci*100)}% (продажи)"
        )
        self.ax_sales.legend(loc="best")

        self.family_cum_rev_line, = self.ax_rev.plot(
            t, cum_rev_det, linestyle="-.", linewidth=2.4,
            label="Моделирование: кумулятивная выручка"
        )
        self.family_cum_rev_fill = self.ax_rev.fill_between(
            t, cum_rev_lo, cum_rev_hi, alpha=0.18,
            label=f"Моделирование: ДИ {int(ci*100)}% (выручка)"
        )
        self.ax_rev.legend(loc="best")

        self.canvas_left.draw_idle()
        self.status_var.set("Моделирование по f(p) выполнено")

    def _remove_family_overlays(self):
        for attr in ("family_cum_sales_line", "family_cum_rev_line"):
            ln = getattr(self, attr, None)
            if ln is not None:
                try: ln.remove()
                except Exception: pass
                setattr(self, attr, None)
        for attr in ("family_cum_sales_fill", "family_cum_rev_fill"):
            poly = getattr(self, attr, None)
            if poly is not None:
                try: poly.remove()
                except Exception: pass
                setattr(self, attr, None)

def main():
    app = StatModelGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
