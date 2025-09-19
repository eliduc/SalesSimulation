# -*- coding: utf-8 -*-
# emulation_uniform.py — v1.6.1
#
# Новое в v1.6.1:
# - «Равномерное вымывание»: изменён алгоритм корректировки цены.
#   Вместо отклонения проданных от равномерных продаж используется
#   отклонение фактического остатка m_k от равномерного остатка R_u.
#   diff_k = (m_k - R_u)/R_u; P_k = P_(k-1) * (1 - k_fb * diff_k).
#   Границы цены и логика последнего интервала («Полные продажи»/«Гарантия») сохранены.
#
# Новое в v1.6.0:
# - Чекбокс «Полные продажи (на последнем шаге)» → «Полные продажи».
# - Поле «Цель P (sold-out к концу)» → «Вероятность sold-out» (диапазон 0.5…0.9999).
# - Добавлен чекбокс «Гарантировать»: при включённых «Полные продажи» и «Гарантировать»
#   и наличии остатков в конце последнего интервала — они досчитываются по минимально
#   допустимой цене типа: (Начальная цена × (1−Диапазон/100)) × Коэф. цены типа.
# - Логика финального интервала обновлена:
#   • Если «Полные продажи» выключен — последний интервал симулируется как обычный
#     (без автоподбора цены и без довыкупа).
#   • Если «Полные продажи» включен — выбирается цена под «Вероятность sold-out».
#   • «Равномерное вымывание» уважает флаг «Полные продажи» на последнем интервале.
#
# Справочно: Новое в v1.5.1:
# - Равномерное вымывание: графики перерисовываются ПОСЛЕ КАЖДОГО шага
#   с явным обновлением UI (update_idletasks()/update()), поэтому линии «растут вправо».
# - Возвращены столбцы `sales_total` и `rev_total` в results.csv/.xlsx — строго справа от блоков
#   `sales_k*` и `rev_k*`. Порядок столбцов в CSV и XLSX синхронизирован.
# - Переименованы вкладки: «Кумул. продажи (все)», «Кумул. продажи (по типам)»,
#   «Кумул. выручка (все)», «Кумул выручка (по типам)».
#
# Справочно: Новое в v1.5.0:
# - Добавлена кнопка «Равномерное вымывание» на вкладке «Симуляция».
#   Автоматически проходит шаги без анимации, корректируя базовую цену на каждом шаге
#   по формуле обратной связи (устаревшая редакция для v1.5.x–v1.6.0).
#
from __future__ import annotations

import json, math, os, threading, queue, csv
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

HAVE_PANDAS = True
try:
    import pandas as pd
except Exception:
    HAVE_PANDAS = False

# --- numba для ускорения «быстрых» симуляций (калибровка/авто-цена) ---
HAVE_NUMBA = False
try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

# ---------------- УТИЛИТЫ ----------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def softmax_vec(u: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = (u - np.max(u)) / max(1e-12, tau)
    e = np.exp(z)
    s = e.sum()
    return e/s if s > 0 and np.isfinite(s) else np.ones_like(u)/len(u)

def pass_prob(price: float, b_lo: float, b_hi: float) -> float:
    if price <= b_lo: return 1.0
    if price >= b_hi: return 0.0
    return (b_hi - price) / (b_hi - b_lo)

# ---------------- СОСТОЯНИЕ ----------------
@dataclass
class AppState:
    total_units: int = 360
    project_days: int = 360
    n_types: int = 4
    price_interval_days: int = 30

    qualities: List[float] = field(default_factory=list)     # доли
    price_coeffs: List[float] = field(default_factory=list)  # доли
    counts: List[int] = field(default_factory=list)

    initial_base_price: float = 1000.0
    price_range_pct: float = 20.0
    price_schedule: List[float] = field(default_factory=list)

    lambda_per_day: float = 6.0
    lambda_growth_pct: float = 0.0
    lambda_trend_type: str = "linear"
    season_amp_pct: float = 0.0
    season_period_days: int = 0
    season_phase_deg: float = 0.0

    budget_low: float = 500.0
    budget_high: float = 2000.0

    mu_a: float = 0.0
    sigma_a: float = 0.30
    mu_b: float = math.log(1.0/1000.0)
    sigma_b: float = 0.40
    softmax_tau: float = 1.0
    uniform_kfb: float = 0.5

    sims: int = 100
    target_sellout_prob: float = 0.95
    full_sell: bool = False      # чекбокс «Полные продажи»
    guarantee: bool = False      # чекбокс «Гарантировать»

# ---------------- БЫСТРАЯ СИМУЛЯЦИЯ ----------------
@njit(cache=True)
def _simulate_once_numba(prices: np.ndarray, qualities: np.ndarray, counts: np.ndarray,
                         lam_by_day: np.ndarray, b_lo: float, b_hi: float,
                         mu_a: float, sg_a: float, mu_b: float, sg_b: float, tau: float):
    K, T = prices.shape
    inv = counts.copy()

    sales = np.zeros(T, np.int64)
    rev = np.zeros(T, np.float64)
    sales_k = np.zeros((K, T), np.int64)
    rev_k = np.zeros((K, T), np.float64)

    remain_total = 0
    for k in range(K): remain_total += inv[k]

    for t in range(T):
        if remain_total <= 0: break
        n = np.random.poisson(lam_by_day[t])
        if n <= 0: continue

        for _ in range(n):
            candidates = np.empty(K, np.int64); m = 0
            for k in range(K):
                if inv[k] <= 0: continue
                p = prices[k, t]
                if p <= b_lo: prob = 1.0
                elif p >= b_hi: prob = 0.0
                else: prob = (b_hi - p) / (b_hi - b_lo)
                if np.random.random() < prob:
                    candidates[m] = k; m += 1
            if m == 0: continue

            a = np.random.lognormal(mu_a, sg_a)
            b = np.random.lognormal(mu_b, sg_b)

            u = np.empty(m+1, np.float64); u[0] = 0.0
            for j in range(m):
                k = candidates[j]
                u[j+1] = a*qualities[k] - b*prices[k, t]

            mx = u[0]
            for j in range(1, m+1):
                if u[j] > mx: mx = u[j]
            ssum = 0.0
            for j in range(m+1):
                val = math.exp((u[j]-mx)/max(1e-12, tau))
                u[j] = val; ssum += val

            r = np.random.random() * ssum
            acc = 0.0; chosen = 0
            for j in range(m+1):
                acc += u[j]
                if r <= acc: chosen = j; break
            if chosen == 0:
                continue

            ksel = candidates[chosen-1]
            if inv[ksel] > 0:
                inv[ksel] -= 1
                remain_total -= 1
                sales[t] += 1; rev[t] += prices[ksel, t]
                sales_k[ksel, t] += 1; rev_k[ksel, t] += prices[ksel, t]

    return sales, rev, sales_k, rev_k

def simulate_once_fast(base_price_by_day: np.ndarray, qualities: np.ndarray,
                       price_coeffs: np.ndarray, counts: np.ndarray,
                       lam_by_day: np.ndarray, b_lo: float, b_hi: float,
                       mu_a: float, sg_a: float, mu_b: float, sg_b: float,
                       tau: float, seed: int):
    prices = (price_coeffs[:, None] * base_price_by_day[None, :]).astype(np.float64)
    qualities = np.asarray(qualities, np.float64)
    counts = np.asarray(counts, np.int64)
    lam_by_day = np.asarray(lam_by_day, np.float64)
    np.random.seed(seed)
    if HAVE_NUMBA:
        return _simulate_once_numba(prices, qualities, counts, lam_by_day,
                                    b_lo, b_hi, mu_a, sg_a, mu_b, sg_b, tau)
    else:
        K, T = prices.shape
        inv = counts.copy()
        sales = np.zeros(T, int); rev = np.zeros(T, float)
        sales_k = np.zeros((K, T), int); rev_k = np.zeros((K, T), float)
        for t in range(T):
            if inv.sum() <= 0: break
            n = np.random.poisson(lam_by_day[t])
            if n <= 0: continue
            for _ in range(n):
                passed = [k for k in range(K) if inv[k]>0 and (np.random.random() < pass_prob(prices[k,t], b_lo, b_hi))]
                if not passed: continue
                a = float(np.random.lognormal(mu_a, sg_a))
                b = float(np.random.lognormal(mu_b, sg_b))
                u = np.array([a*qualities[k]-b*prices[k,t] for k in passed], float)
                u_all = np.concatenate(([0.0], u))
                z = np.exp((u_all - np.max(u_all))/max(1e-12, tau))
                probs = z/z.sum()
                idx = np.random.choice(np.arange(len(u_all)), p=probs)
                if idx == 0: continue
                ksel = passed[idx-1]
                if inv[ksel] > 0:
                    inv[ksel] -= 1
                    sales[t] += 1; rev[t] += prices[ksel, t]
                    sales_k[ksel, t] += 1; rev_k[ksel, t] += prices[ksel, t]
        return sales, rev, sales_k, rev_k

# ---------------- ПРОГРЕСС-ДИАЛОГ ----------------
class ProgressDialog(tk.Toplevel):
    def __init__(self, master, title="Выполняется…", maximum: int = 100):
        super().__init__(master)
        self.title(title); self.resizable(False, False)
        self.grab_set(); self.cancel_callback = None
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        frm = ttk.Frame(self); frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        self.msg = ttk.Label(frm, text="Подготовка…", width=64, anchor="w"); self.msg.pack(fill=tk.X, pady=(0,6))
        self.pb  = ttk.Progressbar(frm, orient=tk.HORIZONTAL, mode="determinate", length=420, maximum=maximum)
        self.pb.pack(fill=tk.X); self.btn = ttk.Button(frm, text="Отмена", command=self._on_cancel)
        self.btn.pack(pady=(10,0)); self.update_idletasks()
        self._center()

    def _center(self):
        m = self.master
        if m:
            x = m.winfo_rootx() + (m.winfo_width()-self.winfo_width())//2
            y = m.winfo_rooty() + (m.winfo_height()-self.winfo_height())//2
            self.geometry(f"+{x}+{y}")

    def set_progress(self, val: int, text: str):
        self.pb["value"] = val; self.msg.config(text=text); self.update_idletasks()

    def set_max(self, maximum: int): self.pb["maximum"] = maximum

    def _on_cancel(self):
        if self.cancel_callback: self.cancel_callback()

# ---------------- ГЛАВНОЕ ПРИЛОЖЕНИЕ ----------------
class RealEstateApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Эмулятор продаж — λ-тренд/сезонность, калибровка, анимация (v1.6.1)")
        self.geometry("1340x940"); self.minsize(1180,760)

        # --- анимация / визуальная часть ---
        self.anim_canvas: Optional[tk.Canvas] = None
        self.show_anim_var = tk.BooleanVar(value=True)
        self.anim_running = False
        self._anim_after_update = None
        self._anim_after_spawn = None
        self.anim_buyers = []
        self.anim_type_slots = []
        self.anim_session = None
        self.anim_speed = 1.75
        self._speed_gain = (3.0/1.75) * 4.0
        self.anim_speed_var = None

        self._cfg_job = None

        # --- история шагов ---
        self.hist_sales_k_steps: List[np.ndarray] = []
        self.hist_rev_k_steps: List[np.ndarray] = []
        self.hist_days_end: List[int] = []
        self.hist_price_points: List[float] = []
        self.remaining_inv: Optional[np.ndarray] = None

        # --- переменные типов ---
        self.type_vars: List[Tuple[tk.DoubleVar, tk.DoubleVar, tk.IntVar]] = []

        # Экспорт сценария
        self.on_export_scenario = self._on_export_scenario

        self.state = AppState()
        self._build_ui()
        self._rebuild_type_rows()
        self._update_allowed_range_label()
        self._load_scenario()
        self._log("Готово. «Сделать шаг» — один интервал. Визуализация — чекбоксом.")

        # калибровка
        self._calib_thread: Optional[threading.Thread] = None
        self._calib_stop = False
        self._calib_queue: "queue.Queue[tuple]" = queue.Queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------
    def _build_ui(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Экспорт сценария (JSON)…", command=self.on_export_scenario)
        filemenu.add_command(label="Сохранить XLSX/CSV…", command=self.on_save_dialog)
        filemenu.add_separator(); filemenu.add_command(label="Сбросить все параметры", command=self.on_reset_all)
        filemenu.add_separator(); filemenu.add_command(label="EXIT", command=self._on_close)
        menubar.add_cascade(label="Файл", menu=filemenu)
        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Калибровать a,b (логнорм.)", command=self.on_calibrate_ab)
        menubar.add_cascade(label="Инструменты", menu=tools)
        self.config(menu=menubar)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL); paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned); right = ttk.Frame(paned)
        paned.add(left, weight=1); paned.add(right, weight=3)

        # левые вкладки
        self.nb_left = ttk.Notebook(left); self.nb_left.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        tab_proj = ttk.Frame(self.nb_left); tab_types = ttk.Frame(self.nb_left)
        tab_buy = ttk.Frame(self.nb_left); tab_sim = ttk.Frame(self.nb_left); tab_log = ttk.Frame(self.nb_left)
        self.nb_left.add(tab_proj, text="Проект"); self.nb_left.add(tab_types, text="Типы")
        self.nb_left.add(tab_buy,  text="Покупатели"); self.nb_left.add(tab_sim, text="Симуляция")
        self.nb_left.add(tab_log,  text="Статус")

        self._build_section_project(tab_proj)
        self._build_section_types(tab_types)
        self._build_section_buyers(tab_buy)
        self._build_section_sim(tab_sim)
        self._build_log(tab_log)

        # правые графики
        self._build_plots(right)

    def _build_section_project(self, parent):
        f = ttk.LabelFrame(parent, text="Параметры проекта"); f.pack(fill=tk.X, padx=8, pady=6)
        self.total_units_var = tk.IntVar(value=self.state.total_units)
        self.project_days_var = tk.IntVar(value=self.state.project_days)
        self.interval_days_var = tk.IntVar(value=self.state.price_interval_days)

        row1 = ttk.Frame(f); row1.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row1, text="Объектов:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row1, textvariable=self.total_units_var, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(row1, text="Горизонт (дни):").grid(row=0, column=2, sticky="w")
        ttk.Entry(row1, textvariable=self.project_days_var, width=8).grid(row=0, column=3, padx=6)
        ttk.Label(row1, text="Интервал цены (дни):").grid(row=0, column=4, sticky="w")
        ttk.Entry(row1, textvariable=self.interval_days_var, width=8).grid(row=0, column=5, padx=6)

        self.initial_price_var = tk.DoubleVar(value=self.state.initial_base_price)
        self.price_range_pct_var = tk.DoubleVar(value=self.state.price_range_pct)

        row2 = ttk.Frame(f); row2.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row2, text="Начальная цена:").grid(row=0, column=0, sticky="w")
        ttk.Entry(row2, textvariable=self.initial_price_var, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(row2, text="Диапазон изменения, %:").grid(row=0, column=2, sticky="w")
        ttk.Entry(row2, textvariable=self.price_range_pct_var, width=6).grid(row=0, column=3, padx=6)

        self.allowed_label = ttk.Label(f, text="", foreground="#555")
        self.allowed_label.pack(fill=tk.X, padx=6, pady=(0,2))

    def _build_section_types(self, parent):
        f = ttk.LabelFrame(parent, text="Типы (качество/коэф. цены/кол-во)")
        f.pack(fill=tk.X, padx=8, pady=6)

        top = ttk.Frame(f); top.pack(fill=tk.X, padx=6, pady=(4,2))
        self.n_types_var = tk.IntVar(value=self.state.n_types)
        ttk.Label(top, text="Число типов (1–5):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(top, from_=1, to=5, textvariable=self.n_types_var, width=5,
                    command=self._rebuild_type_rows).grid(row=0, column=1, padx=6)
        ttk.Button(top, text="Распределить поровну", command=self.on_even_counts).grid(row=0, column=2, padx=6)
        ttk.Button(top, text="Коэф.=100%+(кач-100%)/2", command=self.on_recompute_price_coeffs).grid(row=0, column=3, padx=6)

        hdr = ttk.Frame(f); hdr.pack(fill=tk.X, padx=6, pady=(6,2))
        ttk.Label(hdr, text="Тип").grid(row=0, column=0, sticky="w")
        ttk.Label(hdr, text="Качество, % (80–120)").grid(row=0, column=1, sticky="w")
        ttk.Label(hdr, text="Коэф., %").grid(row=0, column=2, sticky="w")
        ttk.Label(hdr, text="Кол-во").grid(row=0, column=3, sticky="w")

        self.type_rows_container = ttk.Frame(f); self.type_rows_container.pack(fill=tk.X, padx=6, pady=2)
        self.counts_hint = ttk.Label(f, text="", foreground="#555"); self.counts_hint.pack(fill=tk.X, padx=6, pady=2)

    def _build_section_buyers(self, parent):
        f = ttk.LabelFrame(parent, text="Покупатели и поведение")
        f.pack(fill=tk.X, padx=8, pady=6)

        self.lambda_var = tk.DoubleVar(value=self.state.lambda_per_day)
        self.lambda_growth_pct_var = tk.DoubleVar(value=self.state.lambda_growth_pct)
        self.lambda_trend_type_var = tk.StringVar(value=self.state.lambda_trend_type)
        self.season_amp_var = tk.DoubleVar(value=self.state.season_amp_pct)
        self.season_period_var = tk.IntVar(value=self.state.season_period_days)
        self.season_phase_var = tk.DoubleVar(value=self.state.season_phase_deg)
        self.budget_low_var = tk.DoubleVar(value=self.state.budget_low)
        self.budget_high_var = tk.DoubleVar(value=self.state.budget_high)

        row = ttk.Frame(f); row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row, text="λ покупателей/день (старт):").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.lambda_var, width=8).grid(row=0, column=1, padx=6)
        ttk.Label(row, text="Рост λ к концу, % (−50..+200):").grid(row=0, column=2, sticky="w")
        ttk.Entry(row, textvariable=self.lambda_growth_pct_var, width=8).grid(row=0, column=3, padx=6)
        ttk.Label(row, text="Тип тренда:").grid(row=0, column=4, sticky="w")
        ttk.Combobox(row, values=["linear","exp"], textvariable=self.lambda_trend_type_var,
                     width=8, state="readonly").grid(row=0, column=5, padx=6)

        row2 = ttk.Frame(f); row2.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row2, text="Сезонность λ (ампл. %, период дней, фаза°):").grid(row=0, column=0, sticky="w")
        ttk.Entry(row2, textvariable=self.season_amp_var, width=6).grid(row=0, column=1, padx=4)
        ttk.Entry(row2, textvariable=self.season_period_var, width=6).grid(row=0, column=2, padx=4)
        ttk.Entry(row2, textvariable=self.season_phase_var, width=6).grid(row=0, column=3, padx=4)
        ttk.Label(row2, text="Бюджет (ниж., верх.):").grid(row=0, column=4, sticky="w")
        ttk.Entry(row2, textvariable=self.budget_low_var, width=8).grid(row=0, column=5, padx=6)
        ttk.Entry(row2, textvariable=self.budget_high_var, width=8).grid(row=0, column=6, padx=6)

        self.mu_a_var = tk.DoubleVar(value=self.state.mu_a)
        self.sigma_a_var = tk.DoubleVar(value=self.state.sigma_a)
        self.mu_b_var = tk.DoubleVar(value=self.state.mu_b)
        self.sigma_b_var = tk.DoubleVar(value=self.state.sigma_b)
        self.tau_var = tk.DoubleVar(value=self.state.softmax_tau)

        row3 = ttk.Frame(f); row3.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(row3, text="a ~ LogNormal(μa, σa):").grid(row=0, column=0, sticky="w")
        ttk.Entry(row3, textvariable=self.mu_a_var, width=8).grid(row=0, column=1, padx=6)
        ttk.Entry(row3, textvariable=self.sigma_a_var, width=8).grid(row=0, column=2, padx=6)
        ttk.Label(row3, text="b ~ LogNormal(μb, σb):").grid(row=0, column=3, sticky="w")
        ttk.Entry(row3, textvariable=self.mu_b_var, width=8).grid(row=0, column=4, padx=6)
        ttk.Entry(row3, textvariable=self.sigma_b_var, width=8).grid(row=0, column=5, padx=6)
        ttk.Label(row3, text="τ (softmax):").grid(row=0, column=6, sticky="w")
        ttk.Entry(row3, textvariable=self.tau_var, width=8).grid(row=0, column=7, padx=6)

        self.calib_sims_var = tk.IntVar(value=240)
        self.calib_evals_var = tk.IntVar(value=120)
        row4 = ttk.Frame(f); row4.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(row4, text="Калибровать a,b (логнорм.)", command=self.on_calibrate_ab).grid(row=0, column=0, padx=6)
        ttk.Label(row4, text="реплик/точку:").grid(row=0, column=1, sticky="w")
        ttk.Entry(row4, textvariable=self.calib_sims_var, width=8).grid(row=0, column=2, padx=6)
        ttk.Label(row4, text="макс. оценок:").grid(row=0, column=3, sticky="w")
        ttk.Entry(row4, textvariable=self.calib_evals_var, width=8).grid(row=0, column=4, padx=6)

    def _build_section_sim(self, parent):
        f = ttk.LabelFrame(parent, text="Симуляция"); f.pack(fill=tk.X, padx=8, pady=6)

        self.sims_var = tk.IntVar(value=self.state.sims)
        self.target_p_var = tk.DoubleVar(value=self.state.target_sellout_prob)
        self.full_sell_var = tk.BooleanVar(value=self.state.full_sell)
        self.guarantee_var = tk.BooleanVar(value=self.state.guarantee)

        ttk.Label(f, text="Эмуляций для расчётов:").grid(row=0, column=0, sticky="w", padx=2)
        ttk.Entry(f, textvariable=self.sims_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(f, text="Вероятность sold-out:").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Entry(f, textvariable=self.target_p_var, width=8).grid(row=0, column=3, padx=2)
        ttk.Checkbutton(f, text="Полные продажи", variable=self.full_sell_var)\
            .grid(row=0, column=4, padx=2, sticky="w")
        ttk.Checkbutton(f, text="Гарантия", variable=self.guarantee_var)\
            .grid(row=0, column=5, padx=6, sticky="w")

        self.kfb_var = tk.DoubleVar(value=self.state.uniform_kfb)
        ttk.Label(f, text="Коэфф. обратной связи k_fb:").grid(row=1, column=0, sticky="w", padx=6)
        ttk.Entry(f, textvariable=self.kfb_var, width=8).grid(row=1, column=1, padx=6)
        ttk.Button(f, text="Равномерное вымывание", command=self.on_uniform_drain).grid(row=1, column=2, padx=10)
        self._build_section_steps(parent)
        self._build_anim_panel(parent)

    def _build_section_steps(self, parent):
        f = ttk.LabelFrame(parent, text="Шаги (цена постоянна в интервале)")
        f.pack(fill=tk.X, padx=8, pady=6)

        self.current_price_var = tk.DoubleVar(value=self.state.initial_base_price)
        self.allowed_label_step = ttk.Label(f, text="", foreground="#555")
        self.allowed_label_step.grid(row=0, column=0, columnspan=8, sticky="w", padx=6, pady=2)

        ttk.Label(f, text="Цена для шага:").grid(row=1, column=0, sticky="w", padx=6)
        ttk.Entry(f, textvariable=self.current_price_var, width=10).grid(row=1, column=1, padx=6)
        ttk.Button(f, text="Сделать шаг", command=self.on_add_step).grid(row=1, column=2, padx=6)
        ttk.Button(f, text="Отменить шаг", command=self.on_undo_step).grid(row=1, column=3, padx=6)
        ttk.Button(f, text="Начать снова", command=self.on_restart).grid(row=1, column=4, padx=6)
        ttk.Button(f, text="EXIT", command=self._on_close).grid(row=1, column=5, padx=6)

        self.status_summary = ttk.Label(f, text="Статус: нет шагов.", foreground="#444")
        self.status_summary.grid(row=2, column=0, columnspan=8, sticky="w", padx=6, pady=4)

    def _build_anim_panel(self, parent):
        cb_frame = ttk.Frame(parent); cb_frame.pack(fill=tk.X, padx=8, pady=(2,0))
        ttk.Checkbutton(cb_frame, text="Показать анимацию", variable=self.show_anim_var,
                        command=self._toggle_anim_visibility).pack(anchor="w")

        f = ttk.LabelFrame(parent, text="Графическая симуляция (демо)")
        f.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.anim_canvas_holder = f

        ctrl = ttk.Frame(f); ctrl.pack(fill=tk.X, padx=8, pady=(8,0))
        ttk.Label(ctrl, text="Скорость анимации: медленно").pack(side=tk.LEFT)
        self.anim_speed_var = tk.DoubleVar(value=1.75)
        sld = ttk.Scale(ctrl, from_=0.5, to=10.0, orient=tk.HORIZONTAL,
                        variable=self.anim_speed_var, command=self._on_anim_speed_change)
        sld.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        ttk.Label(ctrl, text="быстро").pack(side=tk.LEFT)

        self.anim_canvas = tk.Canvas(f, width=720, height=360,
                                     bg="#f9fbff", highlightthickness=1, highlightbackground="#cbd5e1")
        self.anim_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.anim_canvas.bind("<Configure>", self._on_canvas_configure)

        self._on_anim_speed_change(None)
        self.after(60, self._refresh_anim_scene)

    def _build_log(self, parent):
        lf = ttk.LabelFrame(parent, text="Журнал выполнения")
        lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.log = tk.Text(lf, height=12, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _build_plots(self, parent):
        self.fig_price = Figure(figsize=(8,2.2), dpi=100)
        self.ax_price = self.fig_price.add_subplot(1,1,1)
        self.canvas_price = FigureCanvasTkAgg(self.fig_price, master=parent)
        self.canvas_price.get_tk_widget().pack(fill=tk.X, padx=8, pady=(8,4))

        nb = ttk.Notebook(parent); nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))
        self.nb_plots = nb

        self.tab_sales_agg = ttk.Frame(nb); nb.add(self.tab_sales_agg, text="Кумул. продажи (все)")
        self.fig_sales_agg = Figure(figsize=(8,2.8), dpi=100)
        self.ax_sales_agg = self.fig_sales_agg.add_subplot(1,1,1)
        self.canvas_sales_agg = FigureCanvasTkAgg(self.fig_sales_agg, master=self.tab_sales_agg)
        self.canvas_sales_agg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.tab_sales_types = ttk.Frame(nb); nb.add(self.tab_sales_types, text="Кумул. продажи (по типам)")
        self.fig_sales_types = Figure(figsize=(8,2.8), dpi=100)
        self.ax_sales_types = self.fig_sales_types.add_subplot(1,1,1)
        self.canvas_sales_types = FigureCanvasTkAgg(self.fig_sales_types, master=self.tab_sales_types)
        self.canvas_sales_types.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.tab_rev_agg = ttk.Frame(nb); nb.add(self.tab_rev_agg, text="Кумул. выручка (все)")
        self.fig_rev_agg = Figure(figsize=(8,2.8), dpi=100)
        self.ax_rev_agg = self.fig_rev_agg.add_subplot(1,1,1)
        self.canvas_rev_agg = FigureCanvasTkAgg(self.fig_rev_agg, master=self.tab_rev_agg)
        self.canvas_rev_agg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.tab_rev_types = ttk.Frame(nb); nb.add(self.tab_rev_types, text="Кумул выручка (по типам)")
        self.fig_rev_types = Figure(figsize=(8,2.8), dpi=100)
        self.ax_rev_types = self.fig_rev_types.add_subplot(1,1,1)
        self.canvas_rev_types = FigureCanvasTkAgg(self.fig_rev_types, master=self.tab_rev_types)
        self.canvas_rev_types.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- служебное ----------
    def _log(self, msg: str):
        self.log.insert(tk.END, msg + "\n"); self.log.see(tk.END)

    def _scenario_path(self) -> str:
        try: base = os.path.dirname(os.path.abspath(__file__))
        except NameError: base = os.getcwd()
        return os.path.join(base, "scenario.json")

    def _save_scenario(self, path: Optional[str] = None):
        if path is None: path = self._scenario_path()
        snap = self._snapshot()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
            self._log(f"Сценарий сохранён в {os.path.basename(path)}.")
        except Exception as e:
            self._log(f"Не удалось сохранить сценарий: {e}")

    def _load_scenario(self):
        path = self._scenario_path()
        if not os.path.exists(path): return
        try:
            with open(path, "r", encoding="utf-8") as f:
                sc = json.load(f)
            self.total_units_var.set(sc.get("total_units",  self.state.total_units))
            self.project_days_var.set(sc.get("project_days", self.state.project_days))
            self.interval_days_var.set(sc.get("price_interval_days", self.state.price_interval_days))
            self.initial_price_var.set(sc.get("initial_base_price", self.state.initial_base_price))
            self.price_range_pct_var.set(sc.get("price_range_pct", self.state.price_range_pct))
            self.lambda_var.set(sc.get("lambda_per_day", self.state.lambda_per_day))
            self.lambda_growth_pct_var.set(sc.get("lambda_growth_pct", self.state.lambda_growth_pct))
            self.lambda_trend_type_var.set(sc.get("lambda_trend_type", self.state.lambda_trend_type))
            self.season_amp_var.set(sc.get("season_amp_pct", self.state.season_amp_pct))
            self.season_period_var.set(sc.get("season_period_days", self.state.season_period_days))
            self.season_phase_var.set(sc.get("season_phase_deg", self.state.season_phase_deg))
            self.budget_low_var.set(sc.get("budget_low", self.state.budget_low))
            self.budget_high_var.set(sc.get("budget_high", self.state.budget_high))
            self.mu_a_var.set(sc.get("mu_a", self.state.mu_a))
            self.sigma_a_var.set(sc.get("sigma_a", self.state.sigma_a))
            self.mu_b_var.set(sc.get("mu_b", self.state.mu_b))
            self.sigma_b_var.set(sc.get("sigma_b", self.state.sigma_b))
            self.tau_var.set(sc.get("softmax_tau", self.state.softmax_tau))
            self.sims_var.set(sc.get("sims", self.state.sims))
            self.target_p_var.set(sc.get("target_sellout_prob", self.state.target_sellout_prob))
            self.full_sell_var.set(sc.get("full_sell", self.state.full_sell))
            self.guarantee_var.set(sc.get("guarantee", self.state.guarantee))
            self.kfb_var.set(sc.get("uniform_kfb", self.state.uniform_kfb))
            types = sc.get("types", [])
            if types:
                self.n_types_var.set(len(types)); self._rebuild_type_rows()
                for i, row in enumerate(types):
                    qv, pv, cv = self.type_vars[i]
                    qv.set(float(row.get("quality_pct", 100.0)))
                    pv.set(float(row.get("price_coeff_pct", 100.0)))
                    cv.set(int(row.get("count", 0)))
            self.state.price_schedule = []
            self._update_allowed_range_label()
            self._log(f"Сценарий загружен из {os.path.basename(path)}.")
        except Exception as e:
            self._log(f"Не удалось загрузить сценарий: {e}")

    def _snapshot(self) -> dict:
        types = []
        for i, (qv, pv, cv) in enumerate(self.type_vars, start=1):
            types.append({"index": i, "quality_pct": float(qv.get()),
                          "price_coeff_pct": float(pv.get()), "count": int(cv.get())})
        return {
            "total_units": int(self.total_units_var.get()),
            "project_days": int(self.project_days_var.get()),
            "price_interval_days": int(self.interval_days_var.get()),
            "initial_base_price": float(self.initial_price_var.get()),
            "price_range_pct": float(self.price_range_pct_var.get()),
            "lambda_per_day": float(self.lambda_var.get()),
            "lambda_growth_pct": float(self.lambda_growth_pct_var.get()),
            "lambda_trend_type": str(self.lambda_trend_type_var.get()),
            "season_amp_pct": float(self.season_amp_var.get()),
            "season_period_days": int(self.season_period_var.get()),
            "season_phase_deg": float(self.season_phase_var.get()),
            "budget_low": float(self.budget_low_var.get()),
            "budget_high": float(self.budget_high_var.get()),
            "mu_a": float(self.mu_a_var.get()), "sigma_a": float(self.sigma_a_var.get()),
            "mu_b": float(self.mu_b_var.get()), "sigma_b": float(self.sigma_b_var.get()),
            "softmax_tau": float(self.tau_var.get()),
            "sims": int(self.sims_var.get()),
            "target_sellout_prob": float(self.target_p_var.get()),
            "full_sell": bool(self.full_sell_var.get()),
            "guarantee": bool(self.guarantee_var.get()),
            "uniform_kfb": float(self.kfb_var.get()),
            "types": types,
            "price_schedule": list(self.state.price_schedule),
        }

    def _on_close(self):
        try: self._save_scenario()
        finally: self.destroy()

    # ---------- строки типов ----------
    def _rebuild_type_rows(self):
        if not hasattr(self, "type_rows_container"): return
        for ch in self.type_rows_container.winfo_children(): ch.destroy()
        self.type_vars = []

        n = clamp(int(getattr(self, "n_types_var", tk.IntVar(value=4)).get()), 1, 5)
        total = int(getattr(self, "total_units_var", tk.IntVar(value=360)).get())
        base = total//n; rem = total - base*n

        for i in range(n):
            q_def = clamp(100 + 5*i, 80, 120)
            pc_def = 100 + (q_def-100)/2.0
            c_def = base + (1 if i < rem else 0)

            qv = tk.DoubleVar(value=q_def); pv = tk.DoubleVar(value=pc_def); cv = tk.IntVar(value=c_def)
            self.type_vars.append((qv, pv, cv))
            row = ttk.Frame(self.type_rows_container); row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"Тип {i+1}").grid(row=0, column=0, padx=6, sticky="w")
            ttk.Entry(row, textvariable=qv, width=8).grid(row=0, column=1, padx=6)
            ttk.Entry(row, textvariable=pv, width=10).grid(row=0, column=2, padx=6)
            ttk.Entry(row, textvariable=cv, width=10).grid(row=0, column=3, padx=6)

        self._update_counts_hint()
        self._refresh_anim_scene()

    def _update_counts_hint(self):
        total = int(self.total_units_var.get())
        s = sum(int(cv.get()) for _,_,cv in self.type_vars)
        self.counts_hint.config(text=f"Сумма по типам = {s} (нужно {total}; разница {s-total:+d})")

    # ---------- сбор параметров ----------
    def _collect_state(self) -> Tuple[Optional[AppState], Optional[str]]:
        st = AppState()
        try:
            st.total_units = int(self.total_units_var.get())
            st.project_days = int(self.project_days_var.get())
            st.n_types = int(self.n_types_var.get())
            st.price_interval_days = int(self.interval_days_var.get())
            st.initial_base_price = float(self.initial_price_var.get())
            st.price_range_pct = float(self.price_range_pct_var.get())

            st.lambda_per_day = float(self.lambda_var.get())
            st.lambda_growth_pct = float(self.lambda_growth_pct_var.get())
            st.lambda_trend_type = str(self.lambda_trend_type_var.get())
            st.season_amp_pct = float(self.season_amp_var.get())
            st.season_period_days = int(self.season_period_var.get())
            st.season_phase_deg = float(self.season_phase_var.get())

            st.budget_low = float(self.budget_low_var.get())
            st.budget_high = float(self.budget_high_var.get())

            st.mu_a = float(self.mu_a_var.get()); st.sigma_a = float(self.sigma_a_var.get())
            st.mu_b = float(self.mu_b_var.get()); st.sigma_b = float(self.sigma_b_var.get())
            st.softmax_tau = float(self.tau_var.get())

            st.sims = int(self.sims_var.get())
            st.target_sellout_prob = float(self.target_p_var.get())
            st.full_sell = bool(self.full_sell_var.get())
            st.guarantee = bool(self.guarantee_var.get())
            st.uniform_kfb = float(self.kfb_var.get())
        except Exception:
            return None, "Ошибка ввода. Проверьте числа."

        if st.n_types < 1 or st.n_types > 5: return None, "Число типов 1..5."
        if st.total_units <= 0 or st.project_days <= 0: return None, "N объектов и срок > 0."
        if not (500 <= st.initial_base_price <= 2000): return None, "Начальная цена в [500,2000]."
        if not (1 <= st.price_interval_days <= 360): return None, "Интервал 1..360."
        if st.budget_low >= st.budget_high: return None, "Нижняя граница бюджета должна быть < верхней."
        if st.sims < 100 or st.sims > 100000: return None, "Эмуляций 100..100000."
        if st.sigma_a <= 0 or st.sigma_b <= 0: return None, "σa и σb > 0."
        if st.lambda_growth_pct < -50 or st.lambda_growth_pct > 200: return None, "Рост λ: −50..+200%."
        if st.lambda_trend_type not in ("linear","exp"): return None, "Тип тренда λ: linear или exp."
        if st.season_amp_pct < 0: return None, "Амплитуда сезонности должна быть ≥ 0."
        if not (0.5 <= st.target_sellout_prob <= 0.9999): return None, "Вероятность sold-out — [0.5, 0.9999]."
        if not (0.0 <= st.uniform_kfb <= 1.0): return None, "Коэфф. обратной связи k_fb — [0,1]."

        qs, pcs, cts = [], [], []
        for i, (qv, pv, cv) in enumerate(self.type_vars):
            q = float(qv.get()); pc = float(pv.get()); c = int(cv.get())
            if not (80.0 <= q <= 120.0): return None, f"Качество типа {i+1} — [80,120]."
            qs.append(q/100.0); pcs.append(pc/100.0); cts.append(max(0,c))
        if sum(cts) != st.total_units: return None, "Сумма количеств по типам должна равняться общему N."

        st.qualities, st.price_coeffs, st.counts = qs, pcs, cts
        st.price_schedule = list(self.state.price_schedule)
        return st, None

    # ---------- серии ----------
    def _build_lambda_series(self, st: AppState, T: int, use_growth: bool) -> np.ndarray:
        lam0 = st.lambda_per_day
        if not use_growth or T <= 1:
            lam = np.full(T, lam0, float)
        else:
            g = clamp(st.lambda_growth_pct, -50, 200) / 100.0
            t = np.arange(T, dtype=float)
            if st.lambda_trend_type == "exp":
                lam = lam0 * np.power(1.0 + g, t / max(1.0, T-1.0))
            else:
                lam = lam0 * (1.0 + g * (t / max(1.0, T-1.0)))
        A = st.season_amp_pct / 100.0
        P = st.season_period_days
        if A > 0 and P and P > 0:
            phi = math.radians(st.season_phase_deg)
            t = np.arange(T, dtype=float)
            seasonal = 1.0 + A * np.sin(2.0 * math.pi * t / float(P) + phi)
            lam = lam * np.maximum(0.0, seasonal)
        return lam

    def _build_price_series(self, st: AppState, full_horizon: bool) -> np.ndarray:
        p0 = st.initial_base_price
        if not st.price_schedule:
            return np.full(st.project_days, p0, float) if full_horizon else np.array([], float)
        max_days = st.project_days if full_horizon else min(st.project_days, len(st.price_schedule)*st.price_interval_days)
        series = np.empty(max_days, float)
        for d in range(max_days):
            i = min(d // st.price_interval_days, len(st.price_schedule)-1)
            series[d] = st.price_schedule[i]
        return series

    def _update_allowed_range_label(self):
        p0 = float(self.initial_price_var.get())
        rng = float(self.price_range_pct_var.get())/100.0
        lo, hi = p0*(1-rng), p0*(1+rng)
        txt = f"Допустимый диапазон базовой цены: {lo:.0f} … {hi:.0f}"
        if hasattr(self, "allowed_label"): self.allowed_label.config(text=txt)
        if hasattr(self, "allowed_label_step"): self.allowed_label_step.config(text=txt)

    # ---------- «ГАРАНТИРОВАТЬ» при полных продажах ----------
    def _apply_full_sell_if_needed(self, st: AppState, type_prices_last: np.ndarray,
                                   step_sales_k: np.ndarray, step_rev_k: np.ndarray,
                                   is_last_to_add: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Если включены «Полные продажи» и «Гарантировать», а это последний интервал
        и остались остатки — досчитываем их по минимально допустимой цене типа.
        """
        full_flag = bool(self.full_sell_var.get()) if hasattr(self, "full_sell_var") else st.full_sell
        guarantee_flag = bool(self.guarantee_var.get()) if hasattr(self, "guarantee_var") else st.guarantee

        if not (full_flag and guarantee_flag and is_last_to_add):
            return step_sales_k, step_rev_k
        if self.remaining_inv is None:
            return step_sales_k, step_rev_k

        leftover = np.maximum(self.remaining_inv.astype(int), 0)
        add_total = int(np.sum(leftover))
        if add_total <= 0:
            return step_sales_k, step_rev_k

        # Минимально допустимая базовая цена в пределах коридора
        p0 = float(self.initial_price_var.get()) if hasattr(self, "initial_price_var") else st.initial_base_price
        band = (float(self.price_range_pct_var.get()) if hasattr(self, "price_range_pct_var") else st.price_range_pct)/100.0
        lo_base = max(500.0, p0*(1-band))
        lo_base = float(min(2000.0, lo_base))  # общий коридор базовой цены

        type_min_prices = lo_base * np.array(st.price_coeffs, float)

        step_sales_k = step_sales_k + leftover
        step_rev_k = step_rev_k + type_min_prices * leftover.astype(float)
        self.remaining_inv = self.remaining_inv - leftover
        self._log(f"Гарантировать: добавили {add_total} шт. по минимально допустимой цене типа (база≈{lo_base:.0f}).")
        return step_sales_k, step_rev_k

    # ---------- шаг ----------
    def on_add_step(self):
        st, err = self._collect_state()
        if err: messagebox.showerror("Ошибка параметров", err); return

        max_steps = math.ceil(st.project_days / st.price_interval_days)
        if len(st.price_schedule) >= max_steps:
            self._log("Достигнут конец горизонта. Шаги больше добавлять нельзя.")
            return

        is_last_to_add = (len(st.price_schedule) == max_steps - 1)

        # Выбор цены на шаг:
        if is_last_to_add and bool(self.full_sell_var.get()):
            self._log("Последний шаг (режим «Полные продажи»): авто‑подбор цены под «Вероятность sold‑out»…")
            reps = max(200, int(st.sims))
            price = self._optimize_last_price_for_P(st, P_target=st.target_sellout_prob, reps=reps)
        else:
            price = float(self.current_price_var.get())
            p0 = st.initial_base_price; band = st.price_range_pct/100.0
            lo, hi = max(500.0, p0*(1-band)), min(2000.0, p0*(1+band))
            if not (lo <= price <= hi):
                messagebox.showerror("Ошибка цены", f"Цена шага должна быть в [{lo:.0f}, {hi:.0f}].")
                return

        self.state.price_schedule.append(price)
        st.price_schedule = list(self.state.price_schedule)
        self.current_price_var.set(price)

        step_idx = len(st.price_schedule)
        start_day = (step_idx - 1) * st.price_interval_days
        end_day = min(step_idx * st.price_interval_days, st.project_days)

        lam_series = self._build_lambda_series(st, st.project_days, use_growth=True)
        lam_sum = float(np.sum(lam_series[start_day:end_day]))
        rng = np.random.default_rng(2026 + step_idx*7)
        N_buyers = int(rng.poisson(lam_sum))

        type_prices = price * np.array(st.price_coeffs, float)
        qualities = np.array(st.qualities, float)

        if self.remaining_inv is None:
            self.remaining_inv = np.array([int(cv.get()) for _,_,cv in self.type_vars], int)
        else:
            self.remaining_inv = np.maximum(self.remaining_inv, 0)

        self._log(f"Шаг {step_idx}: цена={price:.0f}. Интервал дней: {start_day+1}–{end_day}. "
                  f"Пуассон: ожид.={lam_sum:.2f}, пришло N={N_buyers}.")

        if not bool(self.show_anim_var.get()):
            step_sales_k = np.zeros(len(type_prices), int)
            for _ in range(N_buyers):
                B = float(rng.uniform(st.budget_low, st.budget_high))
                affordable = [k for k,p in enumerate(type_prices) if self.remaining_inv[k] > 0 and p <= B]
                if not affordable: continue
                a = float(rng.lognormal(st.mu_a, st.sigma_a))
                b = float(rng.lognormal(st.mu_b, st.sigma_b))
                u = np.array([a*qualities[k] - b*type_prices[k] for k in affordable], float)
                u_all = np.concatenate(([0.0], u))
                z = np.exp((u_all - np.max(u_all))/max(1e-12, st.softmax_tau))
                probs = z/z.sum()
                idx = int(rng.choice(np.arange(len(u_all)), p=probs))
                if idx == 0: continue
                ksel = affordable[idx-1]
                if self.remaining_inv[ksel] > 0:
                    self.remaining_inv[ksel] -= 1
                    step_sales_k[ksel] += 1

            step_rev_k = type_prices * step_sales_k.astype(float)
            step_sales_k, step_rev_k = self._apply_full_sell_if_needed(
                st, type_prices, step_sales_k, step_rev_k, is_last_to_add=is_last_to_add
            )

            self._finalize_step_record_and_plot(st, start_day, end_day, price, step_sales_k, step_rev_k,
                                                is_last_to_add)
            return

        # --- режим с анимацией (очередь) ---
        buyers_queue = []
        for _ in range(N_buyers):
            buyers_queue.append(dict(
                B=float(rng.uniform(st.budget_low, st.budget_high)),
                a=float(rng.lognormal(st.mu_a, st.sigma_a)),
                b=float(rng.lognormal(st.mu_b, st.sigma_b))
            ))

        self.anim_session = dict(
            st=st, step_idx=step_idx, start_day=start_day, end_day=end_day,
            price=price, type_prices=type_prices, qualities=qualities,
            buyers_queue=buyers_queue, buyers_total=N_buyers, buyers_spawned=0,
            step_sales_k=np.zeros(len(type_prices), int),
            step_rev_k=np.zeros(len(type_prices), float),
        )

        self._refresh_anim_scene(force=True)
        self.anim_running = True
        self._anim_loop()
        self._anim_spawn_next()

    def on_uniform_drain(self):
        """
        Автоматический проход по интервалам с корректировкой базовой цены
        для равномерного вымывания остатков. Игнорирует чекбокс «Показать анимацию».
        Логика последнего интервала:
        - если «Полные продажи» ВЫКЛ — обрабатывается как обычный шаг (без автоподбора);
        - если «Полные продажи» ВКЛ — используется автоподбор по «Вероятность sold-out».
        """
        # Блокируем, если анимация в работе
        if self.anim_running:
            messagebox.showinfo("Занято", "Сначала дождитесь завершения текущей анимации шага."); return

        # Сбор параметров
        st, err = self._collect_state()
        if err:
            messagebox.showerror("Ошибка параметров", err); return

        # Расчёт максимального числа шагов
        max_steps = math.ceil(st.project_days / st.price_interval_days)
        if len(st.state.price_schedule if hasattr(st, 'state') else self.state.price_schedule) >= max_steps:
            self._log("Равномерное вымывание: интервалов больше нет."); return

        # Подготовка остатков
        if self.remaining_inv is None:
            self.remaining_inv = np.array([int(cv.get()) for _,_,cv in self.type_vars], int)
        else:
            self.remaining_inv = np.maximum(self.remaining_inv, 0)

        # Общие величины
        counts_init = np.array([int(cv.get()) for _,_,cv in self.type_vars], int)
        N_total = int(np.sum(counts_init))
        K_total = max_steps
        k_fb = clamp(float(self.kfb_var.get()), 0.0, 1.0)

        # Сразу обновим подсказку диапазона
        self._update_allowed_range_label()

        # Бежим по оставшимся шагам
        while len(self.state.price_schedule) < max_steps:
            if int(np.sum(self.remaining_inv)) <= 0:
                self._log("Равномерное вымывание: остаток исчерпан, завершаем."); break

            # Актуализируем стейт для вспомогательных процедур
            st, err = self._collect_state()
            if err: messagebox.showerror("Ошибка параметров", err); return
            st.price_schedule = list(self.state.price_schedule)

            step_idx0 = len(st.price_schedule)  # уже выполнено шагов
            is_last_to_add = (step_idx0 == max_steps - 1)

            # Цена текущего шага
            p0 = st.initial_base_price; band = st.price_range_pct/100.0
            lo, hi = max(500.0, p0*(1-band)), min(2000.0, p0*(1+band))

            if step_idx0 == 0:
                # Первый шаг — ровно начальная цена
                price = float(st.initial_base_price)
            elif is_last_to_add and bool(self.full_sell_var.get()):
                # Последний интервал: автоподбор только при «Полные продажи»
                reps = max(200, int(st.sims))
                price = self._optimize_last_price_for_P(st, P_target=st.target_sellout_prob, reps=reps)
            else:
                # Корректировка цены по равномерному вымыванию (по остаткам)
                p_prev = float(self.state.price_schedule[-1]) if step_idx0 > 0 else float(st.initial_base_price)
                k_cur = step_idx0 + 1  # текущий добавляемый шаг (1..K_total)
                m_k = int(np.sum(self.remaining_inv)) if self.remaining_inv is not None else N_total
                R_u = (N_total / float(K_total)) * float(K_total - k_cur + 1)
                if R_u <= 0.0:
                    diff_k = 0.0
                else:
                    diff_k = (m_k - R_u) / R_u
                price = p_prev * (1.0 - k_fb * diff_k)
                price = float(min(max(price, lo), hi))

            # Фиксируем цену в расписание
            self.state.price_schedule.append(float(price))
            st.price_schedule = list(self.state.price_schedule)
            self.current_price_var.set(float(price))

            # Интервал дат для этого шага
            step_idx = len(st.price_schedule)  # номер добавляемого шага (1..K)
            start_day = (step_idx - 1) * st.price_interval_days
            end_day = min(step_idx * st.price_interval_days, st.project_days)

            # Спрос (пуассон по дням)
            lam_series = self._build_lambda_series(st, st.project_days, use_growth=True)
            lam_sum = float(np.sum(lam_series[start_day:end_day]))
            rng = np.random.default_rng(2026 + step_idx*7)
            N_buyers = int(rng.poisson(lam_sum))

            type_prices = float(price) * np.array(st.price_coeffs, float)
            qualities = np.array(st.qualities, float)

            self._log(f"[UNIF] Шаг {step_idx}: цена={price:.0f}. Интервал дней: {start_day+1}–{end_day}. "
                      f"Пуассон: ожид.={lam_sum:.2f}, пришло N={N_buyers}.")

            # Симуляция без анимации
            step_sales_k = np.zeros(len(type_prices), int)
            for _ in range(N_buyers):
                B = float(rng.uniform(st.budget_low, st.budget_high))
                affordable = [k for k,p in enumerate(type_prices) if self.remaining_inv[k] > 0 and p <= B]
                if not affordable: continue
                a = float(rng.lognormal(st.mu_a, st.sigma_a))
                b = float(rng.lognormal(st.mu_b, st.sigma_b))
                u = np.array([a*qualities[k] - b*type_prices[k] for k in affordable], float)
                u_all = np.concatenate(([0.0], u))
                z = np.exp((u_all - np.max(u_all))/max(1e-12, st.softmax_tau))
                probs = z/z.sum()
                idx = int(rng.choice(np.arange(len(u_all)), p=probs))
                if idx == 0: continue
                ksel = affordable[idx-1]
                if self.remaining_inv[ksel] > 0:
                    self.remaining_inv[ksel] -= 1
                    step_sales_k[ksel] += 1

            step_rev_k = type_prices * step_sales_k.astype(float)
            step_sales_k, step_rev_k = self._apply_full_sell_if_needed(
                st, type_prices, step_sales_k, step_rev_k, is_last_to_add=is_last_to_add
            )

            # Финализация
            self._finalize_step_record_and_plot(st, start_day, end_day, float(price),
                                                step_sales_k, step_rev_k, is_last_to_add)

            try:
                self.update_idletasks(); self.update()
            except Exception:
                pass

            if is_last_to_add:
                break

    def on_undo_step(self):
        if not self.state.price_schedule:
            self._log("Нет шагов для отмены."); return
        if self.anim_running or (self.anim_session and self.anim_session["buyers_queue"]):
            messagebox.showinfo("Занято", "Дождитесь завершения текущей анимации шага."); return
        last_price = self.state.price_schedule.pop()
        if self.hist_price_points:
            self.hist_price_points.pop()
            self.hist_days_end.pop()
            step_sales = self.hist_sales_k_steps.pop()
            self.hist_rev_k_steps.pop()
            if self.remaining_inv is not None:
                self.remaining_inv += step_sales.astype(int)
        self._log(f"Шаг отменён. Цена {last_price:.0f} удалена из истории.")
        self._redraw_all_plots_from_history()

    def on_restart(self):
        if self.anim_running:
            messagebox.showinfo("Занято", "Сначала дождитесь завершения текущей анимации шага."); return
        self.state.price_schedule = []
        self.hist_sales_k_steps.clear(); self.hist_rev_k_steps.clear()
        self.hist_days_end.clear(); self.hist_price_points.clear()
        self.remaining_inv = None
        self.current_price_var.set(self.state.initial_base_price)
        self.ax_price.clear(); self.canvas_price.draw()
        self.ax_sales_agg.clear(); self.ax_sales_types.clear()
        self.ax_rev_agg.clear(); self.ax_rev_types.clear()
        self.canvas_sales_agg.draw(); self.canvas_sales_types.draw()
        self.canvas_rev_agg.draw(); self.canvas_rev_types.draw()
        self.status_summary.config(text="Статус: нет шагов.")
        self._log("Сброс истории. Возврат к t=0.")
        self._anim_clear()
        self._refresh_anim_scene(force=True)

    # ---------- авто-цена финального интервала ----------
    def _optimize_last_price_for_P(self, st: AppState, P_target: float, reps: int) -> float:
        p0 = st.initial_base_price; band = st.price_range_pct/100.0
        lo, hi = max(500.0, p0*(1-band)), min(2000.0, p0*(1+band))

        pre = self._build_price_series(st, full_horizon=False)
        T = st.project_days
        remain_days = T - len(pre)
        if remain_days <= 0:
            return float(self.current_price_var.get())

        lam_series = self._build_lambda_series(st, T, use_growth=True)
        q = np.array(st.qualities); pc = np.array(st.price_coeffs); cnt = np.array(st.counts, int)

        def prob(price: float) -> float:
            base = np.concatenate([pre, np.full(remain_days, price, float)])
            success = 0
            for r in range(reps):
                s, _, s_k, _ = simulate_once_fast(base, q, pc, cnt, lam_series,
                                                  st.budget_low, st.budget_high,
                                                  st.mu_a, st.sigma_a, st.mu_b, st.sigma_b,
                                                  st.softmax_tau, seed=987654 + r)
                sold_k = np.sum(s_k, axis=1)
                if np.all(sold_k >= cnt):
                    success += 1
            return success / float(reps)

        p_lo = prob(lo); p_hi = prob(hi)
        self._log(f"Автоподбор финальной цены: P({lo:.0f})≈{p_lo:.3f}, P({hi:.0f})≈{p_hi:.3f}, цель={P_target:.3f}")

        if p_hi >= P_target:
            return hi
        if p_lo < P_target:
            return lo

        L, H = lo, hi; best, bestP = lo, p_lo
        for _ in range(16):
            mid = 0.5*(L+H); pm = prob(mid)
            if pm >= P_target: best, bestP, L = mid, pm, mid
            else: H = mid
        self._log(f"Итог авто-цены финального интервала: {best:.0f} (P≈{bestP:.3f}).")
        return best

    # ---------- графики из истории ----------
    def _redraw_all_plots_from_history(self):
        self.ax_price.clear()
        self.ax_price.plot(self.hist_days_end, self.hist_price_points, marker="o", linewidth=1)
        self.ax_price.set_title("Базовая цена (по интервалам)")
        self.ax_price.set_ylabel("Цена"); self.canvas_price.draw()

        if not self.hist_sales_k_steps:
            for ax in (self.ax_sales_agg, self.ax_sales_types, self.ax_rev_agg, self.ax_rev_types):
                ax.clear()
            self.canvas_sales_agg.draw(); self.canvas_sales_types.draw()
            self.canvas_rev_agg.draw(); self.canvas_rev_types.draw()
            return

        K = len(self.hist_sales_k_steps[0])
        cum_sales_k = np.cumsum(np.stack(self.hist_sales_k_steps, axis=1), axis=1)
        cum_sales_total = np.sum(cum_sales_k, axis=0)
        cum_rev_k = np.cumsum(np.stack(self.hist_rev_k_steps, axis=1), axis=1)
        cum_rev_total = np.sum(cum_rev_k, axis=0)

        x = np.array([0] + self.hist_days_end, float)
        y_sales = np.concatenate(([0.0], cum_sales_total))
        y_rev   = np.concatenate(([0.0], cum_rev_total))

        self.ax_sales_agg.clear(); self.ax_sales_agg.plot(x, y_sales, marker="o", linewidth=1.5)
        self.ax_sales_agg.set_ylabel("Кумулятивные продажи"); self.canvas_sales_agg.draw()

        self.ax_rev_agg.clear(); self.ax_rev_agg.plot(x, y_rev, marker="o", linewidth=1.5)
        self.ax_rev_agg.set_ylabel("Кумулятивная выручка"); self.canvas_rev_agg.draw()

        self.ax_sales_types.clear()
        for k in range(K):
            yk = np.concatenate(([0.0], cum_sales_k[k]))
            self.ax_sales_types.plot(x, yk, marker="o", linewidth=1.2, label=f"Тип {k+1}")
        if K > 1: self.ax_sales_types.legend(loc="upper left", fontsize=8)
        self.ax_sales_types.set_ylabel("Кумулятивные продажи"); self.canvas_sales_types.draw()

        self.ax_rev_types.clear()
        for k in range(K):
            yk = np.concatenate(([0.0], cum_rev_k[k]))
            self.ax_rev_types.plot(x, yk, marker="o", linewidth=1.2, label=f"Тип {k+1}")
        if K > 1: self.ax_rev_types.legend(loc="upper left", fontsize=8)
        self.ax_rev_types.set_ylabel("Кумулятивная выручка"); self.ax_rev_types.set_xlabel("День")
        self.canvas_rev_types.draw()

        steps_done = len(self.state.price_schedule)
        max_steps = math.ceil(int(self.project_days_var.get()) / int(self.interval_days_var.get()))
        last_rem = int(np.sum(self.remaining_inv)) if self.remaining_inv is not None else 0
        self.status_summary.config(
            text=f"Статус: шаг {steps_done}/{max_steps} • Кум. продажи: {int(y_sales[-1])} • Остаток≈{last_rem}"
        )

    # ---------- Canvas: реакция на размер ----------
    def _on_canvas_configure(self, event):
        if self._cfg_job is not None:
            try: self.after_cancel(self._cfg_job)
            except Exception: pass
        self._cfg_job = self.after(30, lambda: self._refresh_anim_scene(force=True))

    # ---------- отрисовка сцены ----------
    def _refresh_anim_scene(self, force: bool=False):
        if not self.show_anim_var.get() or self.anim_canvas is None:
            return
        c = self.anim_canvas
        W = c.winfo_width(); H = c.winfo_height()
        if W < 200:
            try: W = int(c["width"])
            except Exception: W = 720
        if H < 120:
            try: H = int(c["height"])
            except Exception: H = 360

        c.delete("all")
        margin = 18

        # Постоянная легенда (левый верх) — полные подписи
        lx = margin; ly = margin
        c.create_rectangle(lx, ly, lx+10, ly+10, outline="", fill="#0ea5e9", tags=("legend",))
        c.create_text(lx+14, ly+6, anchor="w", text="Бюджет", font=("Segoe UI", 8),
                      fill="#0ea5e9", tags=("legend",))
        ly += 14
        c.create_rectangle(lx, ly, lx+10, ly+10, outline="", fill="#22c55e", tags=("legend",))
        c.create_text(lx+14, ly+6, anchor="w", text="Важность качества", font=("Segoe UI", 8),
                      fill="#22c55e", tags=("legend",))
        ly += 14
        c.create_rectangle(lx, ly, lx+10, ly+10, outline="", fill="#f59e0b", tags=("legend",))
        c.create_text(lx+14, ly+6, anchor="w", text="Важность цены", font=("Segoe UI", 8),
                      fill="#f59e0b", tags=("legend",))

        # Нижняя светлая полоса
        lane_height = 56
        lane_y = H - lane_height
        c.create_rectangle(0, lane_y, W, H, fill="#eef2ff", outline="", tags=("bg",))

        if len(self.type_vars) == 0:
            return

        # Витрины справа
        area_x1 = int(W * 0.46); area_x2 = W - margin
        slot_w = int((area_x2 - area_x1) * 0.86)
        slot_x = int(area_x1 + (area_x2 - area_x1 - slot_w) / 2)

        # Центр принятия решений
        free_left = margin
        free_right = slot_x - 10
        if free_right < free_left + 40:
            free_right = free_left + 40
        dec_x = int((free_left + free_right) / 2)
        dec_y = int(H / 2)

        # Увеличенный ЦПР — центр остаётся тем же
        box_w = 220
        box_h = 56
        half_w = box_w // 2
        dec_x = int(clamp(dec_x, free_left + half_w, free_right - half_w))

        c.create_rectangle(dec_x - half_w, dec_y - box_h//2,
                           dec_x + half_w, dec_y + box_h//2,
                           outline="#cbd5e1", fill="#ffffff", tags=("decision",))
        c.create_text(dec_x, dec_y - box_h//2 - 30,
                      text="Центр принятия решения", font=("Segoe UI", 11, "bold"),
                      fill="#475569", anchor="s")
        c.create_oval(dec_x-7, dec_y-7, dec_x+7, dec_y+7, outline="#94a3b8", fill="#94a3b8")
        c.create_text(dec_x, dec_y-14, text="●", fill="#64748b")

        # Витрины типов
        K = len(self.type_vars)
        gap = 10
        total_h = H - 2*margin
        slot_h = int((total_h - gap*(K-1)) / max(1, K))
        self.anim_type_slots = []

        if self.remaining_inv is None:
            inv_now = np.array([int(cv.get()) for _,_,cv in self.type_vars], int)
        else:
            inv_now = self.remaining_inv.copy()

        base_price = float(self.current_price_var.get() if self.state.price_schedule == [] else self.state.price_schedule[-1])
        prices = [base_price * (float(pv.get())/100.0) for _, pv, _ in self.type_vars]
        qualities = [float(qv.get())/100.0 for qv, _, _ in self.type_vars]

        for i in range(K):
            y1 = int(margin + i*(slot_h+gap)); y2 = y1 + slot_h
            c.create_rectangle(slot_x, y1, slot_x+slot_w, y2, outline="#94a3b8", width=1.0, fill="#ffffff",
                               tags=(f"type{i}", "type_box"))
            c.create_text(slot_x+8, y1+12, anchor="w",
                          text=f"Тип {i+1} | цена≈{int(prices[i])} | кач: {int(round(qualities[i]*100))}%",
                          font=("Segoe UI", 9), fill="#334155", tags=(f"type{i}",))
            bar_w = slot_w - 16; bar_x = slot_x+8; bar_y = y1 + 22
            qv = clamp(qualities[i], 0.6, 1.2)
            fill = "#22c55e" if qv >= 1.05 else ("#a3e635" if qv >= 0.95 else "#f59e0b")
            c.create_rectangle(bar_x, bar_y, bar_x + int(bar_w*qv/1.2), bar_y+8, outline="", fill=fill, tags=(f"type{i}",))
            c.create_text(slot_x+slot_w-8, y1+12, anchor="e", text=f"осталось: {int(inv_now[i])}",
                          font=("Segoe UI", 9, "bold"), fill="#0f172a",
                          tags=(f"type{i}", f"type{i}_counter"))
            self.anim_type_slots.append((slot_x + 20, y1 + slot_h/2))

        self._anim_decision_center = (dec_x, dec_y)

    # ---------- цикл анимации ----------
    def _anim_spawn_next(self):
        if not self.anim_running or self.anim_session is None:
            return
        if any(b.get("state") in ("approach", "think") for b in self.anim_buyers):
            return

        queue = self.anim_session["buyers_queue"]
        if not queue:
            if not self.anim_buyers:
                self._finish_animation_step()
            return

        buyer = queue.pop(0)
        self.anim_session["buyers_spawned"] += 1

        c = self.anim_canvas
        if c is None: return
        W = c.winfo_width() or 720; H = c.winfo_height() or 360
        dec_x, dec_y = getattr(self, "_anim_decision_center", (int(W*0.3), int(H*0.5)))

        x0 = 10.0; y0 = float(dec_y)

        color = np.random.choice(["#ef4444","#84cc16","#22c55e","#06b6d4","#3b82f6","#a855f7","#f97316"])
        tag = f"buyer{np.random.randint(10**9)}"

        c.create_line(x0, y0-10, x0, y0, width=3, fill=color, tags=(tag,"buyer"))
        c.create_line(x0, y0, x0-7, y0+10, width=2, fill=color, tags=(tag,"buyer"))
        c.create_line(x0, y0, x0+7, y0+10, width=2, fill=color, tags=(tag,"buyer"))
        c.create_line(x0, y0-6, x0-8, y0, width=2, fill=color, tags=(tag,"buyer"))
        c.create_line(x0, y0-6, x0+8, y0, width=2, fill=color, tags=(tag,"buyer"))
        c.create_oval(x0-6, y0-20, x0+6, y0-8, outline=color, width=2, tags=(tag,"buyer"))

        buyer_state = dict(
            tag=tag, x=float(x0), y=float(y0),
            state="approach", think_timer=int(round(500/30)),
            bubble_ids=[], bars_ids=[], after_ids=[],
            B=float(buyer["B"]), a=float(buyer["a"]), b=float(buyer["b"]),
            target_k=None
        )
        self.anim_buyers.append(buyer_state)

    def _anim_loop(self):
        if not self.anim_running: return
        if self.anim_canvas is None: return
        c = self.anim_canvas
        W = c.winfo_width() or 720; H = c.winfo_height() or 360
        dec_x, dec_y = getattr(self, "_anim_decision_center", (int(W*0.3), int(H*0.5)))
        remove = []
        eff = self.anim_speed * self._speed_gain

        for buyer in list(self.anim_buyers):
            tag = buyer["tag"]; x, y = buyer["x"], buyer["y"]
            state = buyer["state"]

            if state == "approach":
                dx = min(2.4 * eff, dec_x - x)
                if dx < 0: dx = 0
                c.move(tag, dx, 0); buyer["x"] = x + dx
                if abs(buyer["x"] - dec_x) < 1.5:
                    buyer["state"] = "think"; buyer["think_timer"] = int(round(500/30))
                    self._anim_begin_thinking(buyer, dec_x, dec_y)

            elif state == "think":
                buyer["think_timer"] -= 1
                if buyer["think_timer"] <= 0:
                    self._cancel_buyer_after_jobs(buyer)
                    for bid in buyer.get("bubble_ids", []):
                        try: c.delete(bid)
                        except Exception: pass
                    buyer["bubble_ids"] = []
                    for bid in buyer.get("bars_ids", []):
                        try: c.delete(bid)
                        except Exception: pass
                    buyer["bars_ids"] = []

                    sess = self.anim_session; st = sess["st"]
                    prices = sess["type_prices"]; qualities = sess["qualities"]
                    affordable = [k for k,p in enumerate(prices)
                                  if (self.remaining_inv is not None and self.remaining_inv[k] > 0) and (p <= buyer["B"])]
                    if not affordable:
                        buyer["state"] = "leave"
                    else:
                        u = np.array([buyer["a"]*qualities[k] - buyer["b"]*prices[k] for k in affordable], float)
                        u_all = np.concatenate(([0.0], u))
                        z = np.exp((u_all - np.max(u_all))/max(1e-12, float(self.tau_var.get())))
                        probs = z/z.sum()
                        idx = int(np.random.choice(np.arange(len(u_all)), p=probs))
                        if idx == 0:
                            buyer["state"] = "leave"
                        else:
                            buyer["target_k"] = affordable[idx-1]
                            buyer["state"] = "move_vert"

                    self._anim_spawn_next()

            elif state == "leave":
                dx = -2.0 * eff
                c.move(tag, dx, 0); buyer["x"] = x + dx
                if buyer["x"] < -40: remove.append(buyer)

            elif state == "move_vert":
                k = buyer["target_k"]
                if k is None or k >= len(self.anim_type_slots):
                    buyer["state"] = "leave"; continue
                tx, ty = self.anim_type_slots[k]
                step = 1.8 * eff
                dy = step if ty > y else -step
                if abs(ty - y) < step:
                    buyer["state"] = "move_horiz"
                else:
                    c.move(tag, 0, dy); buyer["y"] = y + dy

            elif state == "move_horiz":
                k = buyer["target_k"]
                if k is None or k >= len(self.anim_type_slots):
                    buyer["state"] = "leave"; continue
                tx, ty = self.anim_type_slots[k]
                step = 2.2 * eff
                dx = step if tx > x else -step
                if abs(tx - x) < step:
                    if self.remaining_inv is not None and self.remaining_inv[k] > 0:
                        self.remaining_inv[k] -= 1
                        for item in c.find_withtag(f"type{k}_counter"):
                            c.itemconfigure(item, text=f"осталось: {int(self.remaining_inv[k])}")
                        price_k = self.anim_session["type_prices"][k]
                        self.anim_session["step_sales_k"][k] += 1
                        self.anim_session["step_rev_k"][k] += price_k
                    remove.append(buyer)
                else:
                    c.move(tag, dx, 0); buyer["x"] = x + dx

        for b in remove:
            try: self.anim_canvas.delete(b["tag"])
            except Exception: pass
            if b in self.anim_buyers:
                self.anim_buyers.remove(b)

        self._anim_after_update = self.after(25, self._anim_loop)

        if self.anim_session and not self.anim_session["buyers_queue"] and not self.anim_buyers:
            self._finish_animation_step()

    def _finish_animation_step(self):
        if not self.anim_session: return
        sess = self.anim_session
        st = sess["st"]
        step_sales_k = sess["step_sales_k"].astype(int)
        step_rev_k = sess["step_rev_k"].astype(float)

        is_last = (len(st.price_schedule) == math.ceil(st.project_days/st.price_interval_days))
        step_sales_k, step_rev_k = self._apply_full_sell_if_needed(
            st, sess["type_prices"], step_sales_k, step_rev_k, is_last_to_add=is_last
        )

        self._anim_stop()

        self._finalize_step_record_and_plot(st, sess["start_day"], sess["end_day"], sess["price"],
                                            step_sales_k, step_rev_k, is_last)
        self.anim_session = None
        self._refresh_anim_scene(force=True)

    def _anim_stop(self):
        self.anim_running = False
        for b in list(self.anim_buyers):
            self._cancel_buyer_after_jobs(b)
        if self._anim_after_update is not None:
            try: self.after_cancel(self._anim_after_update)
            except Exception: pass
            self._anim_after_update = None
        if self._anim_after_spawn is not None:
            try: self.after_cancel(self._anim_after_spawn)
            except Exception: pass
            self._anim_after_spawn = None

    def _anim_clear(self):
        if self.anim_canvas: self.anim_canvas.delete("all")
        self.anim_buyers.clear(); self.anim_type_slots.clear()
        self.anim_session = None

    # ---------- калибровка (фон) ----------
    def on_calibrate_ab(self):
        st, err = self._collect_state()
        if err: messagebox.showerror("Ошибка параметров", err); return
        series = self._build_price_series(st, full_horizon=True)
        if series.size == 0:
            messagebox.showerror("Нет цен", "Сначала задайте хотя бы один шаг или начальную цену.")
            return

        max_evals = max(20, int(self.calib_evals_var.get()))
        reps = max(100, int(self.calib_sims_var.get()))
        dlg = ProgressDialog(self, title="Калибровка a,b (логнорм.)", maximum=max_evals)
        self._calib_stop = False
        dlg.cancel_callback = lambda: setattr(self, "_calib_stop", True)
        self._calib_queue = queue.Queue()

        th = threading.Thread(target=self._calib_worker,
                              args=(st, series, reps, max_evals, self._calib_queue), daemon=True)
        self._calib_thread = th; th.start()

        def poll():
            try:
                while True:
                    msg = self._calib_queue.get_nowait()
                    kind = msg[0]
                    if kind == "progress":
                        payload = msg[1]
                        if isinstance(payload, tuple) and len(payload) >= 3:
                            evals_done, bestJ, best_params = payload[:3]
                            dlg.set_progress(int(evals_done),
                                f"Оценка {int(evals_done)}/{max_evals} | J={bestJ:.4g} | "
                                f"μa={best_params[0]:.3f}, σa={best_params[1]:.3f}, "
                                f"μb={best_params[2]:.3f}, σb={best_params[3]:.3f}")
                        else:
                            dlg.set_progress(int(dlg.pb['value']), "Идёт калибровка…")
                    elif kind == "done":
                        ok, best_params, text = msg[1]
                        dlg.destroy()
                        if ok and best_params:
                            self.mu_a_var.set(best_params[0]); self.sigma_a_var.set(best_params[1])
                            self.mu_b_var.set(best_params[2]); self.sigma_b_var.set(best_params[3])
                            self._save_scenario()
                            self._log("Калибровка завершена: " + text)
                        else:
                            self._log("Калибровка прервана: " + text)
                        return
            except queue.Empty:
                pass
            self.after(120, poll)
        poll()

    def _calib_worker(self, st: AppState, series: np.ndarray, reps: int, max_evals: int,
                      out_queue: "queue.Queue[tuple]"):
        T = len(series); N = st.total_units
        rng = np.random.default_rng(2027)
        lam_const = np.full(T, st.lambda_per_day, float)

        def eval_one(mu_a, sg_a, mu_b, sg_b):
            ts, sold_share_T = [], []
            q = np.array(st.qualities); pc = np.array(st.price_coeffs); cnt = np.array(st.counts)
            for r in range(reps):
                s, _, s_k, _ = simulate_once_fast(series, q, pc, cnt, lam_const,
                                                  st.budget_low, st.budget_high,
                                                  mu_a, sg_a, mu_b, sg_b, st.softmax_tau, seed=98765+r)
                cum = np.cumsum(s)
                sold_share_T.append(float(cum[-1]) / N)
                if cum[-1] >= N:
                    idx = int(np.argmax(cum >= N)); ts.append(idx+1)
                else:
                    w = min(30, T); last = cum[-1]; prev = cum[-w] if T>w else 0.0
                    rate = (last-prev)/max(1,w); extra = (N-last)/max(rate, 1e-8)
                    ts.append(T + float(extra))
            mT, vT, sT = float(np.mean(ts)), float(np.var(ts)), float(np.mean(sold_share_T))
            J = ((mT - T)/T)**2 + 0.25*max(0.0, 1.0 - sT)**2 + 0.05*(vT/(T*T))
            return J, mT, sT

        p0 = st.initial_base_price
        mu_a_lo, mu_a_hi = math.log(0.2), math.log(5.0)
        sg_a_lo, sg_a_hi = 0.05, 1.00
        mu_b_lo, mu_b_hi = math.log(1.0/(5.0*p0)), math.log(1.0/(0.1*p0))
        sg_b_lo, sg_b_hi = 0.05, 1.50

        best = None; evals = 0
        n_init = max(20, max_evals//3)
        for _ in range(n_init):
            if getattr(self, "_calib_stop", False):
                out_queue.put(("done", (False, None, "Отменено пользователем"))); return
            cand = (rng.uniform(mu_a_lo, mu_a_hi), rng.uniform(sg_a_lo, sg_a_hi),
                    rng.uniform(mu_b_lo, mu_b_hi), rng.uniform(sg_b_lo, sg_b_hi))
            J, mT, sT = eval_one(*cand); evals += 1
            if best is None or J < best[0]: best = (J, *cand, mT, sT)
            out_queue.put(("progress", (evals, best[0], (best[1],best[2],best[3],best[4]))))
            if evals >= max_evals: break

        step = np.array([0.4,0.25,0.4,0.4])
        while evals < max_evals and not getattr(self, "_calib_stop", False):
            _, mu_a, sg_a, mu_b, sg_b, mT, sT = best
            for _ in range(8):
                if evals >= max_evals or getattr(self, "_calib_stop", False): break
                cand = np.array([mu_a, sg_a, mu_b, sg_b]) + rng.normal(0, step, size=4)
                cand[0] = clamp(cand[0], mu_a_lo, mu_a_hi); cand[1] = clamp(cand[1], sg_a_lo, sg_a_hi)
                cand[2] = clamp(cand[2], mu_b_lo, mu_b_hi); cand[3] = clamp(cand[3], sg_b_lo, sg_b_hi)
                J, mT, sT = eval_one(cand[0], cand[1], cand[2], cand[3]); evals += 1
                if J < best[0]: best = (J, cand[0], cand[1], cand[2], cand[3], mT, sT)
                out_queue.put(("progress", (evals, best[0], (best[1],best[2],best[3],best[4]))))
                if evals >= max_evals: break
            step *= 0.6

        if getattr(self, "_calib_stop", False):
            out_queue.put(("done", (False, None, "Отменено пользователем"))); return
        J, mu_a, sg_a, mu_b, sg_b, mT, sT = best
        text = (f"μa={mu_a:.3f}, σa={sg_a:.3f}, μb={mu_b:.3f}, σb={sg_b:.3f}; "
                f"ср.день распродажи≈{mT:.1f}/{T}, доля к T≈{sT:.3f}, J={J:.4g}")
        out_queue.put(("done", (True, (float(mu_a), float(sg_a), float(mu_b), float(sg_b)), text)))

    # ---------- меню/сервис ----------
    def on_save_dialog(self):
        if not os.path.exists("results.csv"):
            messagebox.showinfo("Нет данных", "Файлы появятся после хотя бы одного шага."); return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx" if HAVE_PANDAS else ".csv",
                                            filetypes=[("Excel","*.xlsx"),("CSV","*.csv")] if HAVE_PANDAS else [("CSV","*.csv")],
                                            initialfile="results.xlsx" if HAVE_PANDAS else "results.csv")
        if not path: return
        try:
            src = "results.xlsx" if path.lower().endswith(".xlsx") and os.path.exists("results.xlsx") else "results.csv"
            with open(src,"rb") as fsrc, open(path,"wb") as fdst: fdst.write(fsrc.read())
            self._log(f"Скопировано: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def on_even_counts(self):
        try: total = int(self.total_units_var.get())
        except Exception: total = 0
        n = len(self.type_vars); base = total//n; rem = total - base*n
        for i, (_,_,cv) in enumerate(self.type_vars):
            cv.set(base + (1 if i < rem else 0))
        self._update_counts_hint(); self._refresh_anim_scene(force=True)

    def on_recompute_price_coeffs(self):
        for qv, pv, _ in self.type_vars:
            q = float(qv.get()); pv.set(100.0 + (q-100.0)/2.0)
        self._refresh_anim_scene(force=True)

    def on_reset_all(self):
        if self.anim_running:
            messagebox.showinfo("Занято", "Сначала дождитесь завершения текущей анимации шага."); return
        self.state = AppState()
        self.total_units_var.set(self.state.total_units)
        self.project_days_var.set(self.state.project_days)
        self.interval_days_var.set(self.state.price_interval_days)
        self.initial_price_var.set(self.state.initial_base_price)
        self.price_range_pct_var.set(self.state.price_range_pct)
        self.lambda_var.set(self.state.lambda_per_day)
        self.lambda_growth_pct_var.set(self.state.lambda_growth_pct)
        self.lambda_trend_type_var.set(self.state.lambda_trend_type)
        self.season_amp_var.set(self.state.season_amp_pct)
        self.season_period_var.set(self.state.season_period_days)
        self.season_phase_var.set(self.state.season_phase_deg)
        self.budget_low_var.set(self.state.budget_low)
        self.budget_high_var.set(self.state.budget_high)
        self.mu_a_var.set(self.state.mu_a); self.sigma_a_var.set(self.state.sigma_a)
        self.mu_b_var.set(self.state.mu_b); self.sigma_b_var.set(self.state.sigma_b)
        self.tau_var.set(self.state.softmax_tau)
        self.sims_var.set(self.state.sims)
        self.target_p_var.set(self.state.target_sellout_prob)
        self.full_sell_var.set(self.state.full_sell)
        self.guarantee_var.set(self.state.guarantee)
        self.n_types_var.set(self.state.n_types)
        self._rebuild_type_rows()
        self.on_restart()

    def _on_export_scenario(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON","*.json")],
                                            initialfile="scenario.json")
        if not path: return
        self._save_scenario(path)

    # ---------- ДОБАВЛЕНО ранее: управление визуализацией ----------
    def _toggle_anim_visibility(self):
        if not bool(self.show_anim_var.get()):
            self._anim_stop()
            if self.anim_canvas:
                try: self.anim_canvas.delete("all")
                except Exception: pass
        self._refresh_anim_scene(force=True)

    def _on_anim_speed_change(self, _event):
        try:
            self.anim_speed = float(self.anim_speed_var.get())
        except Exception:
            self.anim_speed = 1.75

    def _finalize_step_record_and_plot(self, st: AppState, start_day: int, end_day: int, price: float,
                                       step_sales_k: np.ndarray, step_rev_k: np.ndarray,
                                       is_last_to_add: bool):
        # 1) фиксируем шаг в истории
        self.hist_sales_k_steps.append(np.array(step_sales_k, int))
        self.hist_rev_k_steps.append(np.array(step_rev_k, float))
        self.hist_days_end.append(int(end_day))
        self.hist_price_points.append(float(price))

        # 2) перерисовываем графики
        self._redraw_all_plots_from_history()

        # 3) экспорт результатов (CSV + XLSX c листами Sales и Settings)
        try:
            K = len(self.type_vars)
            S = len(self.hist_sales_k_steps)
            interval = int(self.interval_days_var.get())
            counts_init = np.array([int(cv.get()) for _, _, cv in self.type_vars], int)

            sales_steps = np.stack(self.hist_sales_k_steps, axis=0)   # (S, K)
            rev_steps   = np.stack(self.hist_rev_k_steps,   axis=0)   # (S, K)
            cum_sales   = np.cumsum(sales_steps, axis=0)              # (S, K)
            cum_rev     = np.cumsum(rev_steps,   axis=0)              # (S, K)
            remaining   = counts_init[None, :] - cum_sales            # (S, K)

            rows: list[dict] = []
            for s in range(S):
                row: dict = {
                    "step": s + 1,
                    "start_day": s * interval + 1,
                    "end_day":   min((s + 1) * interval, int(self.project_days_var.get())),
                    "base_price": float(self.state.price_schedule[s]),
                }

                # продажи по типам + кумулятивы (cum сразу после sales_k*)
                for k in range(K):
                    sk  = int(sales_steps[s, k])
                    csk = int(cum_sales[s, k])
                    row[f"sales_k{k+1}"]     = sk
                    row[f"cum_sales_k{k+1}"] = csk

                # totals по продажам
                stot  = int(np.sum(sales_steps[s, :]))
                cstot = int(np.sum(cum_sales[s, :]))
                row["sales_total"]     = stot
                row["cum_sales_total"] = cstot

                # выручка по типам + кумулятивы (cum сразу после rev_k*)
                for k in range(K):
                    rk  = float(rev_steps[s, k])
                    crk = float(cum_rev[s, k])
                    row[f"rev_k{k+1}"]     = rk
                    row[f"cum_rev_k{k+1}"] = crk

                # totals по выручке
                rtot  = float(np.sum(rev_steps[s, :]))
                crtot = float(np.sum(cum_rev[s, :]))
                row["rev_total"]     = rtot
                row["cum_rev_total"] = crtot

                # остатки по типам
                for k in range(K):
                    row[f"remaining_k{k+1}"] = int(remaining[s, k])

                rows.append(row)

            # порядок колонок = порядок добавления ключей
            fieldnames = list(rows[0].keys())

            # CSV (как раньше) — всегда пишем
            with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

            # XLSX: пытаемся записать Sales + Settings, но без жёсткой привязки к xlsxwriter
            if HAVE_PANDAS:
                import pandas as pd

                # Sales
                df_sales = pd.DataFrame(rows)
                df_sales = df_sales[fieldnames]

                # Settings из снапшота сценария
                snap = self._snapshot()
                settings_items: list[tuple[str, object]] = []

                for key in [
                    "total_units", "project_days", "price_interval_days",
                    "initial_base_price", "price_range_pct",
                    "lambda_per_day", "lambda_growth_pct", "lambda_trend_type",
                    "season_amp_pct", "season_period_days", "season_phase_deg",
                    "budget_low", "budget_high",
                    "mu_a", "sigma_a", "mu_b", "sigma_b",
                    "softmax_tau",
                    "sims", "target_sellout_prob", "full_sell", "uniform_kfb"
                ]:
                    if key in snap:
                        settings_items.append((key, snap[key]))

                # price_schedule
                if "price_schedule" in snap:
                    settings_items.append((
                        "price_schedule",
                        ", ".join(f"{p:.2f}" for p in snap["price_schedule"])
                    ))

                # types
                for t in snap.get("types", []):
                    idx = t.get("index", "?")
                    settings_items.append((f"type{idx}_quality_pct",     t.get("quality_pct")))
                    settings_items.append((f"type{idx}_price_coeff_pct", t.get("price_coeff_pct")))
                    settings_items.append((f"type{idx}_count",           t.get("count")))

                df_settings = pd.DataFrame(settings_items, columns=["name", "value"])

                # Пытаемся сначала с openpyxl, затем с xlsxwriter. Если оба недоступны — создаём settings.csv
                wrote_xlsx = False
                for engine in (None, "openpyxl", "xlsxwriter"):
                    try:
                        if engine is None:
                            # Пускай pandas сам выберет доступный движок
                            with pd.ExcelWriter("results.xlsx") as writer:
                                df_sales.to_excel(writer, sheet_name="Sales", index=False)
                                df_settings.to_excel(writer, sheet_name="Settings", index=False)
                        else:
                            with pd.ExcelWriter("results.xlsx", engine=engine) as writer:
                                df_sales.to_excel(writer, sheet_name="Sales", index=False)
                                df_settings.to_excel(writer, sheet_name="Settings", index=False)
                        wrote_xlsx = True
                        break
                    except Exception as _e:
                        wrote_xlsx = False

                if not wrote_xlsx:
                    # резерв: отдельный CSV для настроек, чтобы данные не потерялись
                    df_settings.to_csv("settings.csv", index=False, encoding="utf-8-sig")
                    self._log("Предупреждение: не удалось создать results.xlsx (нет openpyxl/xlsxwriter). "
                              "Записал settings.csv рядом с results.csv.")
        except Exception as e:
            self._log(f"Предупреждение: не удалось сохранить результаты: {e}")

        # 4) pop-up по завершении всех шагов
        if is_last_to_add:
            try:
                messagebox.showinfo("Интервалы завершены",
                                    "Интервал продаж завершён: все шаги выполнены.")
            except Exception:
                pass

    # ---------- ПОМОЩНИКИ «баббла» ----------
    def _animate_bar_growth(self, item_id: int, x1: float, by: float, x2: float,
                            target_h: float, duration_ms: int = 160, steps: int = 8):
        if self.anim_canvas is None: return
        c = self.anim_canvas
        steps = max(1, int(steps))
        duration_ms = max(1, int(duration_ms))
        for i in range(steps + 1):
            t = int(i * duration_ms / steps)
            def _step(i=i):
                h = target_h * (i / steps)
                try:
                    c.coords(item_id, x1, by - h, x2, by)
                except Exception:
                    pass
            self.after(t, _step)

    def _cancel_buyer_after_jobs(self, buyer: dict):
        ids = list(buyer.get("after_ids", []))
        buyer["after_ids"] = []
        for aid in ids:
            try: self.after_cancel(aid)
            except Exception: pass

    def _anim_begin_thinking(self, buyer: dict, dec_x: int, dec_y: int):
        if self.anim_canvas is None: return
        c = self.anim_canvas
        hx, hy = dec_x, dec_y - 14

        bar_w, bar_h = 8, 28
        bx0 = dec_x + 34
        by0 = dec_y + 8

        st, _ = self._collect_state()
        if st is None:
            b_lo, b_hi = 500.0, 2000.0
            p0 = 1000.0
        else:
            b_lo, b_hi = st.budget_low, st.budget_high
            p0 = st.initial_base_price

        def _scale(val, vmin, vmax):
            return max(4, min(bar_h, (val - vmin)/max(1e-9, vmax - vmin)*bar_h))

        target_budget   = _scale(buyer["B"], b_lo, b_hi)
        target_quality  = _scale(buyer["a"], 0.2, 5.0)
        target_priceimp = _scale(buyer["b"], 1.0/(5.0*p0), 1.0/(0.1*p0))

        t2 = c.create_oval(hx+8, hy-2, hx+16, hy+6,
                           outline="#475569", width=1.0, fill="#ffffff",
                           tags=(buyer["tag"],"buyer"))
        buyer["bubble_ids"].append(t2)
        id1 = c.create_rectangle(bx0, by0, bx0+bar_w, by0, outline="", fill="#0ea5e9",
                                 tags=(buyer["tag"],"buyer"))
        buyer["bars_ids"].append(id1)
        self._animate_bar_growth(id1, bx0, by0, bx0+bar_w, target_budget, duration_ms=160, steps=8)

        def stage2():
            if self.anim_canvas is None: return
            try:
                t1 = c.create_oval(hx+16, hy-12, hx+28, hy,
                                   outline="#475569", width=1.0, fill="#ffffff",
                                   tags=(buyer["tag"],"buyer"))
                buyer["bubble_ids"].append(t1)
                x1 = bx0 + bar_w + 2
                id2 = c.create_rectangle(x1, by0, x1+bar_w, by0, outline="", fill="#22c55e",
                                         tags=(buyer["tag"],"buyer"))
                buyer["bars_ids"].append(id2)
                self._animate_bar_growth(id2, x1, by0, x1+bar_w, target_quality, duration_ms=160, steps=8)
            except Exception:
                pass

        def stage3():
            if self.anim_canvas is None: return
            try:
                main = c.create_oval(hx+26, hy-36, hx+86, hy+4,
                                     outline="#475569", width=1.5, fill="#ffffff",
                                     tags=(buyer["tag"],"buyer"))
                buyer["bubble_ids"].append(main)
                x1 = bx0 + 2*(bar_w + 2)
                id3 = c.create_rectangle(x1, by0, x1+bar_w, by0, outline="", fill="#f59e0b",
                                         tags=(buyer["tag"],"buyer"))
                buyer["bars_ids"].append(id3)
                self._animate_bar_growth(id3, x1, by0, x1+bar_w, target_priceimp, duration_ms=160, steps=8)
            except Exception:
                pass

        a2 = self.after(170, stage2)
        a3 = self.after(340, stage3)
        buyer["after_ids"].extend([a2, a3])

# ---------- запуск ----------
if __name__ == "__main__":
    app = RealEstateApp()
    app.mainloop()
