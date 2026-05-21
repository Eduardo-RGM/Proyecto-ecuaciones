"""
╔══════════════════════════════════════════════════════════════════════════╗
║   SOLUCIONADOR DE ECUACIONES DIFERENCIALES HOMOGÉNEAS                   ║
║   Proyecto Integrador — Ingeniería en Sistemas Computacionales          ║
║   Versión 3.0 — Con teclado matemático y problema de aplicación real    ║
╠══════════════════════════════════════════════════════════════════════════╣
║   Aplicación 1 (Tipo 1): Flujo de refrigerante en Data Center           ║
║   Aplicación 2 (Tipo 2): Vibración de HDD en rack de servidores         ║
╠══════════════════════════════════════════════════════════════════════════╣
║   Instalación:  pip install customtkinter matplotlib sympy numpy        ║
║   Ejecución:    python solucionador_homogeneas.py                       ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
import warnings; warnings.filterwarnings("ignore")
import customtkinter as ctk
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch

# ─── Símbolos ────────────────────────────────────────────────────────────────
xs = sp.Symbol("x", real=True)
ys = sp.Symbol("y", real=True)
vs = sp.Symbol("v", real=True)
ts = sp.Symbol("t", positive=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Paleta ──────────────────────────────────────────────────────────────────
BG    = "#0b0b18"
PANEL = "#11112a"
CARD  = "#181835"
INP   = "#0e0e24"
ACC   = "#7c6fff"
CYAN  = "#4cc9f0"
GREEN = "#7bed9f"
YEL   = "#ffd166"
RED   = "#f72585"
DIM   = "#55557a"
MID   = "#8888b8"
MAIN  = "#e0e0f8"
SEP   = "#232348"
PAL   = ["#f72585","#4cc9f0","#7bed9f","#ffd166","#c77dff","#ff9f43"]

# ─── Datos del teclado ───────────────────────────────────────────────────────
# (botón, interno_sympy, display_unicode, tipo)
VARS = [
    ("x",   "x",       "x",    "var"),
    ("y",   "y",       "y",    "var"),
    ("x²",  "x**2",   "x²",   "var"),
    ("y²",  "y**2",   "y²",   "var"),
    ("xy",  "x*y",    "xy",   "var"),
    ("x³",  "x**3",   "x³",   "var"),
    ("x²y", "x**2*y", "x²y",  "var"),
    ("xy²", "x*y**2", "xy²",  "var"),
    ("y³",  "y**3",   "y³",   "var"),
    ("(",   "(",       "(",    "pop"),
    (")",   ")",       ")",    "pcl"),
]

NUM_ROWS = [
    [("7","7","7","d"),("8","8","8","d"),("9","9","9","d"),("+","+","+","op"),("-","-","−","op")],
    [("4","4","4","d"),("5","5","5","d"),("6","6","6","d"),(".",".",".","d"), ("⌫","","","back")],
    [("1","1","1","d"),("2","2","2","d"),("3","3","3","d"),("±","","","neg"), ("C","","","clr")],
    [("  0  ","0","0","d")],
]


# ═══════════════════════════════════════════════════════════════════════════════
#  WIDGET: Campo de expresión con tokens
# ═══════════════════════════════════════════════════════════════════════════════
class ExprField(ctk.CTkFrame):
    def __init__(self, master, label="", placeholder="", **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self._tok   = []            # [(interno, display, tipo)]
        self._ph    = placeholder

        if label:
            ctk.CTkLabel(self, text=label,
                         font=ctk.CTkFont("Segoe UI", 11),
                         text_color=MID).grid(row=0, column=0, sticky="w", padx=2)

        self._box = ctk.CTkFrame(self, fg_color=INP, corner_radius=8,
                                  border_width=2, border_color=SEP)
        self._box.grid(row=1, column=0, sticky="ew", padx=2, pady=(2,4))
        self._box.grid_columnconfigure(0, weight=1)

        self._lbl = ctk.CTkLabel(self._box, text="",
                                  font=ctk.CTkFont("Courier New", 15),
                                  text_color=MAIN, anchor="w")
        self._lbl.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        self._cur = ctk.CTkLabel(self._box, text="",
                                  font=ctk.CTkFont("Courier New", 16),
                                  text_color=ACC)
        self._cur.grid(row=0, column=1, padx=(0, 8))

        for w in (self._box, self._lbl, self._cur):
            w.bind("<Button-1>", lambda e: self.event_generate("<<Click>>"))

        self._refresh()

    def activate(self, on: bool):
        self._box.configure(border_color=ACC if on else SEP)
        self._cur.configure(text="│" if on else "")

    def push(self, internal: str, display: str, kind: str):
        t = self._tok
        if t:
            lt = t[-1][2]
            # Fusión de dígitos
            if kind == "d" and lt == "d":
                t[-1] = (t[-1][0]+internal, t[-1][1]+display, "d")
                self._refresh(); return
            # Multiplicación implícita
            need = (
                (kind in ("var","pop")) and lt in ("d","var","pcl") or
                (kind == "d") and lt in ("var","pcl")
            )
            if need:
                t.append(("*","·","_m"))
        t.append((internal, display, kind))
        self._refresh()

    def backspace(self):
        if not self._tok: return
        last = self._tok[-1]
        if last[2] == "d" and len(last[0]) > 1:
            self._tok[-1] = (last[0][:-1], last[1][:-1], "d")
        else:
            self._tok.pop()
            if self._tok and self._tok[-1][2] == "_m":
                self._tok.pop()
        self._refresh()

    def clear(self):
        self._tok.clear()
        self._refresh()

    def negate(self):
        if self._tok and self._tok[0][2] == "_neg":
            self._tok.pop(0)
        else:
            self._tok.insert(0, ("-","−","_neg"))
        self._refresh()

    def set_tokens(self, tok_list):
        """Carga lista de tokens directamente: [(interno, display, tipo)]"""
        self._tok = list(tok_list)
        self._refresh()

    def get_expr(self) -> str: return "".join(t[0] for t in self._tok)
    def get_disp(self) -> str: return "".join(t[1] for t in self._tok)

    def _refresh(self):
        d = self.get_disp()
        self._lbl.configure(text=d if d else self._ph,
                             text_color=MAIN if d else DIM)


# ═══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Solucionador — Ecuaciones Diferenciales Homogéneas")
        self.geometry("1440x880")
        self.minsize(1200, 780)
        self.configure(fg_color=BG)
        self._act   = None   # ExprField activo
        self._m_val = ctk.DoubleVar(value=0.5)
        self._k_val = ctk.DoubleVar(value=100.0)
        self._c_val = ctk.DoubleVar(value=4.0)
        self._build()

    # ══════════════════════════════════════════════════════════════════════════
    #  LAYOUT PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════
    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Panel izquierdo (scrollable)
        lo = ctk.CTkFrame(self, width=460, corner_radius=0, fg_color=PANEL)
        lo.grid(row=0, column=0, sticky="nsew")
        lo.grid_propagate(False)
        lo.grid_rowconfigure(0, weight=1)
        lo.grid_columnconfigure(0, weight=1)

        lscroll = ctk.CTkScrollableFrame(lo, fg_color="transparent",
                                          scrollbar_button_color=SEP,
                                          scrollbar_button_hover_color=ACC)
        lscroll.grid(row=0, column=0, sticky="nsew")
        lscroll.grid_columnconfigure(0, weight=1)
        self._left(lscroll)

        # Panel derecho
        right = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self._right(right)

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL IZQUIERDO
    # ══════════════════════════════════════════════════════════════════════════
    def _left(self, p):
        def sep(r):
            ctk.CTkFrame(p, height=1, fg_color=SEP).grid(
                row=r, column=0, padx=16, pady=6, sticky="ew")

        # ── Encabezado ───────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(p, fg_color=CARD, corner_radius=12)
        hdr.grid(row=0, column=0, padx=14, pady=(16,8), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="∂  Solucionador de ED Homogéneas",
                     font=ctk.CTkFont("Segoe UI", 14, "bold"),
                     text_color=MAIN).grid(row=0, column=0, pady=10, padx=14, sticky="w")
        ctk.CTkLabel(hdr, text="Ingeniería en Sistemas Computacionales",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=DIM).grid(row=1, column=0, pady=(0,10), padx=14, sticky="w")

        # ── Selector de tipo ─────────────────────────────────────────────────
        ctk.CTkLabel(p, text="  Selecciona el tipo de ecuación:",
                     font=ctk.CTkFont("Segoe UI", 11), text_color=DIM
                     ).grid(row=1, column=0, padx=16, pady=(8,4), sticky="w")

        self.eq_type = ctk.StringVar(value="tipo2")
        rf = ctk.CTkFrame(p, fg_color=CARD, corner_radius=10)
        rf.grid(row=2, column=0, padx=14, pady=(0,6), sticky="ew")
        rf.grid_columnconfigure((0,1), weight=1)

        for col, (val, lbl, sub) in enumerate([
            ("tipo1", "Tipo 1", "1° orden  ·  y = vx"),
            ("tipo2", "Tipo 2", "2° orden  ·  ec. caract."),
        ]):
            f = ctk.CTkFrame(rf, fg_color="transparent")
            f.grid(row=0, column=col, padx=8, pady=10, sticky="ew")
            ctk.CTkRadioButton(f, text=lbl, variable=self.eq_type, value=val,
                                command=self._on_type,
                                font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                fg_color=ACC).pack(anchor="w")
            ctk.CTkLabel(f, text=sub, font=ctk.CTkFont("Courier New", 10),
                          text_color=DIM).pack(anchor="w")

        sep(3)

        # ── Área de inputs ───────────────────────────────────────────────────
        self.inp = ctk.CTkFrame(p, fg_color="transparent")
        self.inp.grid(row=4, column=0, padx=14, pady=4, sticky="ew")
        self.inp.grid_columnconfigure(0, weight=1)

        sep(5)

        # ── Teclado matemático ───────────────────────────────────────────────
        ctk.CTkLabel(p, text="  Teclado matemático",
                     font=ctk.CTkFont("Segoe UI", 11), text_color=DIM
                     ).grid(row=6, column=0, padx=16, pady=(4,4), sticky="w")

        self.kbd = ctk.CTkFrame(p, fg_color=CARD, corner_radius=12)
        self.kbd.grid(row=7, column=0, padx=14, pady=4, sticky="ew")
        self.kbd.grid_columnconfigure(0, weight=1)

        sep(8)

        # ── Botón resolver ───────────────────────────────────────────────────
        self.btn_s = ctk.CTkButton(
            p, text="▶   Resolver paso a paso",
            font=ctk.CTkFont("Segoe UI", 14, "bold"), height=52,
            corner_radius=12, fg_color=ACC, hover_color="#6655ee",
            command=self._solve)
        self.btn_s.grid(row=9, column=0, padx=14, pady=(8,6), sticky="ew")

        ctk.CTkButton(p, text="✕  Limpiar todo",
                       height=36, corner_radius=8,
                       fg_color="transparent", border_width=1,
                       border_color=SEP, hover_color=CARD, text_color=MID,
                       command=self._clear
                       ).grid(row=10, column=0, padx=14, pady=(0,12), sticky="ew")

        self.stlbl = ctk.CTkLabel(p, text="",
                                   font=ctk.CTkFont("Segoe UI", 10),
                                   text_color=DIM, wraplength=420)
        self.stlbl.grid(row=11, column=0, padx=16, pady=4, sticky="w")

        # Construir estado inicial
        self._build_t2()
        self._build_kbd()

    # ══════════════════════════════════════════════════════════════════════════
    #  INPUTS DINÁMICOS
    # ══════════════════════════════════════════════════════════════════════════
    def _clr_inp(self):
        for w in self.inp.winfo_children(): w.destroy()
        self._act = None

    def _activate(self, f: ExprField):
        if self._act and self._act is not f:
            self._act.activate(False)
        self._act = f
        f.activate(True)

    # ── Tipo 1 ───────────────────────────────────────────────────────────────
    def _build_t1(self):
        self._clr_inp()
        f = self.inp

        ctk.CTkLabel(f, text="  Forma:  M(x,y)·dx + N(x,y)·dy = 0",
                     font=ctk.CTkFont("Courier New", 11),
                     text_color=CYAN).grid(row=0, column=0, sticky="w", padx=2, pady=(4,6))

        ctk.CTkLabel(f,
                     text="  ① Haz clic en el campo a llenar   ② Usa el teclado de abajo",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=DIM).grid(row=1, column=0, sticky="w", padx=2, pady=(0,8))

        self.fm = ExprField(f, label="  M(x,y) =", placeholder="clic aquí y usa el teclado")
        self.fm.grid(row=2, column=0, sticky="ew")
        self.fm.bind("<<Click>>", lambda e: self._activate(self.fm))

        self.fn = ExprField(f, label="  N(x,y) =", placeholder="clic aquí y usa el teclado")
        self.fn.grid(row=3, column=0, sticky="ew")
        self.fn.bind("<<Click>>", lambda e: self._activate(self.fn))

        ctk.CTkLabel(f, text="  Ejemplos rápidos:",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=4, column=0, sticky="w", padx=2, pady=(10,4))

        exs = [
            ("x² + y²,  −2xy",
             [("x**2","x²","var"),("+","+","op"),("y**2","y²","var")],
             [("-","−","op"),("2","2","d"),("x","x","var"),("y","y","var")]),
            ("2xy,  x² − y²",
             [("2","2","d"),("x","x","var"),("y","y","var")],
             [("x**2","x²","var"),("-","−","op"),("y**2","y²","var")]),
            ("x² + xy,  y²",
             [("x**2","x²","var"),("+","+","op"),("x","x","var"),("y","y","var")],
             [("y**2","y²","var")]),
            ("x³ + y³,  x²y",
             [("x**3","x³","var"),("+","+","op"),("y**3","y³","var")],
             [("x**2*y","x²y","var")]),
        ]
        for i,(lbl,mt,nt) in enumerate(exs):
            ctk.CTkButton(f, text=lbl, height=30, corner_radius=6,
                           fg_color=CARD, hover_color=SEP,
                           font=ctk.CTkFont("Courier New", 11), text_color=MID,
                           command=lambda mt=mt, nt=nt: self._fill1(mt, nt)
                           ).grid(row=5+i, column=0, padx=2, pady=2, sticky="ew")

        self._activate(self.fm)

    def _fill1(self, mt, nt):
        self.fm.set_tokens(mt)
        self.fn.set_tokens(nt)

    # ── Tipo 2 ───────────────────────────────────────────────────────────────
    def _build_t2(self):
        self._clr_inp()
        f = self.inp

        ctk.CTkLabel(f, text="  Forma:  a·y'' + b·y' + c·y = 0",
                     font=ctk.CTkFont("Courier New", 11),
                     text_color=CYAN).grid(row=0, column=0, sticky="w", padx=2, pady=(4,6))

        ctk.CTkLabel(f,
                     text="  ① Haz clic en el coeficiente   ② Usa el teclado de abajo",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=DIM).grid(row=1, column=0, sticky="w", padx=2, pady=(0,8))

        cf = ctk.CTkFrame(f, fg_color="transparent")
        cf.grid(row=2, column=0, sticky="ew")
        cf.grid_columnconfigure((0,1,2), weight=1)

        self.t2f = {}
        for col,(lbl,sub,ph) in enumerate([
            ("a","(y'')", "ej: 2"),
            ("b","(y')", "ej: -6"),
            ("c","(y)", "ej: 9"),
        ]):
            fr = ctk.CTkFrame(cf, fg_color=CARD, corner_radius=8)
            fr.grid(row=0, column=col, padx=4, pady=4, sticky="ew")
            fr.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(fr, text=f"  {lbl}",
                          font=ctk.CTkFont("Segoe UI", 16, "bold"),
                          text_color=ACC).grid(row=0, column=0, sticky="w", padx=8, pady=(6,0))
            ctk.CTkLabel(fr, text=f"  {sub}",
                          font=ctk.CTkFont("Segoe UI", 9),
                          text_color=DIM).grid(row=1, column=0, sticky="w", padx=8)
            ef = ExprField(fr, placeholder=ph)
            ef.grid(row=2, column=0, sticky="ew", padx=6, pady=(4,8))
            ef.bind("<<Click>>", lambda e, ef=ef: self._activate(ef))
            self.t2f[lbl] = ef

        ctk.CTkLabel(f, text="  Ejemplos rápidos:",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=3, column=0, sticky="w", padx=2, pady=(10,4))

        exs = [
            ("y'' − 5y' + 6y = 0  (raíces reales)",    1,-5, 6),
            ("y'' − 6y' + 9y = 0  (raíz repetida)",     1,-6, 9),
            ("y'' + 4y = 0        (complejas puras)",    1, 0, 4),
            ("y'' + 2y' + 5y = 0  (complejas)",          1, 2, 5),
        ]
        for i,(lbl,a,b,c) in enumerate(exs):
            ctk.CTkButton(f, text=lbl, height=30, corner_radius=6,
                           fg_color=CARD, hover_color=SEP,
                           font=ctk.CTkFont("Courier New", 11), text_color=MID,
                           command=lambda a=a,b=b,c=c: self._fill2(a,b,c)
                           ).grid(row=4+i, column=0, padx=2, pady=2, sticky="ew")

        self._activate(self.t2f["a"])

    def _fill2(self, a, b, c):
        for k,v in [("a",a),("b",b),("c",c)]:
            toks = []
            if v < 0:
                toks.append(("-","−","_neg"))
                toks.append((str(abs(int(v))),str(abs(int(v))),"d"))
            elif v != 0:
                toks.append((str(int(v)),str(int(v)),"d"))
            self.t2f[k].set_tokens(toks)

    def _on_type(self):
        if self.eq_type.get() == "tipo1":
            self._build_t1()
        else:
            self._build_t2()
        self._build_kbd()
        self._update_app_tab()

    # ══════════════════════════════════════════════════════════════════════════
    #  TECLADO MATEMÁTICO (estilo GeoGebra)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_kbd(self):
        for w in self.kbd.winfo_children(): w.destroy()
        f = self.kbd
        tipo = self.eq_type.get()
        row = 0

        def _btn(parent, label, cmd, r, c, fg=CARD, hv=SEP,
                  tc=MAIN, bold=False, span=1, h=40):
            font = ctk.CTkFont("Segoe UI", 14, "bold" if bold else "normal")
            ctk.CTkButton(parent, text=label, font=font,
                           height=h, corner_radius=8,
                           fg_color=fg, hover_color=hv, text_color=tc,
                           command=cmd
                           ).grid(row=r, column=c, columnspan=span,
                                  padx=3, pady=3, sticky="ew")

        # ── Variables (solo Tipo 1) ───────────────────────────────────────────
        if tipo == "tipo1":
            ctk.CTkLabel(f, text="  Variables y términos",
                         font=ctk.CTkFont("Segoe UI", 10, "bold"),
                         text_color=CYAN
                         ).grid(row=row, column=0, pady=(10,4), padx=10, sticky="w")
            row += 1

            vf = ctk.CTkFrame(f, fg_color="#0e1a2e", corner_radius=8)
            vf.grid(row=row, column=0, padx=8, pady=(0,6), sticky="ew")
            for i in range(6): vf.grid_columnconfigure(i, weight=1)

            for idx,(lbl,intl,disp,kind) in enumerate(VARS):
                r2, c2 = divmod(idx, 6)
                ek = {"var":"var","pop":"pop","pcl":"pcl"}.get(kind,kind)
                _btn(vf, lbl,
                      lambda i=intl,d=disp,k=ek: self._push(i,d,k),
                      r2, c2, fg="#0e1a2e", hv="#1a2a4a", tc=CYAN, bold=True, h=38)
            row += 1

            ctk.CTkFrame(f, height=1, fg_color=SEP).grid(
                row=row, column=0, padx=10, pady=4, sticky="ew")
            row += 1

        # ── Números y operadores ──────────────────────────────────────────────
        if tipo == "tipo1":
            ctk.CTkLabel(f, text="  Números y operadores",
                         font=ctk.CTkFont("Segoe UI", 10, "bold"),
                         text_color=MID
                         ).grid(row=row, column=0, pady=(4,4), padx=10, sticky="w")
        else:
            ctk.CTkLabel(f, text="  Ingresa el coeficiente (haz clic en a, b o c primero)",
                         font=ctk.CTkFont("Segoe UI", 10),
                         text_color=DIM
                         ).grid(row=row, column=0, pady=(10,4), padx=10, sticky="w")
        row += 1

        nf = ctk.CTkFrame(f, fg_color="transparent")
        nf.grid(row=row, column=0, padx=8, pady=(0,10), sticky="ew")
        for i in range(5): nf.grid_columnconfigure(i, weight=1)

        for ri, rowdat in enumerate(NUM_ROWS):
            for ci,(lbl,intl,disp,kind) in enumerate(rowdat):
                if kind == "back":
                    _btn(nf, "⌫  Borrar", self._back,
                          ri, ci, fg="#2a1020", hv="#4a1030", tc="#ff88aa", h=40)
                elif kind == "neg":
                    _btn(nf, "±  Negar", self._neg,
                          ri, ci, fg=CARD, hv=SEP, tc=MID, h=40)
                elif kind == "clr":
                    _btn(nf, "✕  Limpiar", self._clrf,
                          ri, ci, fg="#2a1020", hv="#4a1030", tc="#ff88aa", h=40)
                elif kind == "op":
                    ek = "op"
                    _btn(nf, lbl, lambda i=intl,d=disp,k=ek: self._push(i,d,k),
                          ri, ci, fg="#0e2a1e", hv="#1a4a2e", tc=GREEN, bold=True, h=40)
                elif kind in ("d","pop","pcl"):
                    ek = {"d":"d","pop":"pop","pcl":"pcl"}[kind]
                    _btn(nf, lbl, lambda i=intl,d=disp,k=ek: self._push(i,d,k),
                          ri, ci, fg=CARD, hv=SEP, tc=MAIN, h=40)

    def _push(self, i, d, k):
        if self._act: self._act.push(i, d, k)
    def _back(self):
        if self._act: self._act.backspace()
    def _neg(self):
        if self._act: self._act.negate()
    def _clrf(self):
        if self._act: self._act.clear()

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL DERECHO — 3 PESTAÑAS
    # ══════════════════════════════════════════════════════════════════════════
    def _right(self, p):
        ctk.CTkLabel(p,
                     text="Solucionador de Ecuaciones Diferenciales Homogéneas",
                     font=ctk.CTkFont("Segoe UI", 18, "bold"),
                     text_color=MAIN).grid(row=0, column=0, pady=(0,12), sticky="w")

        self.tabs = ctk.CTkTabview(p, anchor="nw",
                                    segmented_button_fg_color=CARD,
                                    segmented_button_selected_color=ACC,
                                    segmented_button_selected_hover_color="#6655ee",
                                    fg_color=CARD)
        self.tabs.grid(row=1, column=0, sticky="nsew")

        self.tab_app  = self.tabs.add("🔬  Problema de aplicación")
        self.tab_proc = self.tabs.add("📋  Procedimiento paso a paso")
        self.tab_graf = self.tabs.add("📈  Gráfica de solución")

        for t in (self.tab_app, self.tab_proc, self.tab_graf):
            t.grid_columnconfigure(0, weight=1)
            t.grid_rowconfigure(0, weight=1)

        # Textbox procedimiento
        self.out = ctk.CTkTextbox(self.tab_proc,
                                   font=ctk.CTkFont("Courier New", 13),
                                   wrap="word", corner_radius=8,
                                   fg_color=INP, text_color=MAIN,
                                   scrollbar_button_color=SEP)
        self.out.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Gráfica
        self.fig_g = Figure(figsize=(7,5), dpi=100)
        self.fig_g.patch.set_facecolor(INP)
        self.cv_g = FigureCanvasTkAgg(self.fig_g, master=self.tab_graf)
        self.cv_g.get_tk_widget().configure(bg=INP)
        self.cv_g.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Pestaña aplicación (se construye después)
        self.app_frame = ctk.CTkScrollableFrame(self.tab_app, fg_color="transparent",
                                                  scrollbar_button_color=SEP)
        self.app_frame.grid(row=0, column=0, sticky="nsew")
        self.app_frame.grid_columnconfigure(0, weight=1)

        self._welcome()
        self._blank_graph()
        self._update_app_tab()

    # ══════════════════════════════════════════════════════════════════════════
    #  PESTAÑA DE PROBLEMA DE APLICACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    def _update_app_tab(self):
        for w in self.app_frame.winfo_children(): w.destroy()
        if self.eq_type.get() == "tipo1":
            self._build_app_t1()
        else:
            self._build_app_t2()

    def _section(self, parent, title, row):
        f = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        f.grid(row=row, column=0, padx=6, pady=6, sticky="ew")
        f.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f, text=title,
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color=ACC).grid(row=0, column=0, padx=14, pady=(10,4), sticky="w")
        return f

    def _body(self, parent, text, row, tc=MID):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=tc, justify="left",
                     wraplength=680).grid(row=row, column=0, padx=14, pady=(2,8), sticky="w")

    # ── Aplicación Tipo 1: Flujo en enfriamiento de data center ──────────────
    def _build_app_t1(self):
        af = self.app_frame

        # Título
        ttl = ctk.CTkFrame(af, fg_color="#0e1a2e", corner_radius=12)
        ttl.grid(row=0, column=0, padx=6, pady=(6,4), sticky="ew")
        ttl.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ttl, text="🖥️  Aplicación: Flujo de refrigerante en Data Center",
                     font=ctk.CTkFont("Segoe UI", 14, "bold"),
                     text_color=CYAN).grid(row=0, column=0, padx=14, pady=(12,4), sticky="w")
        ctk.CTkLabel(ttl,
                     text="Ecuaciones diferenciales homogéneas en ingeniería de sistemas de enfriamiento",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=1, column=0, padx=14, pady=(0,12), sticky="w")

        # Diagrama
        fig_app = Figure(figsize=(7, 2.8), dpi=96)
        fig_app.patch.set_facecolor(INP)
        ax = fig_app.add_subplot(111)
        self._draw_datacenter(ax)
        cv = FigureCanvasTkAgg(fig_app, master=af)
        cv.get_tk_widget().configure(bg=INP)
        cv.get_tk_widget().grid(row=1, column=0, padx=6, pady=4, sticky="ew")
        cv.draw()

        # Contexto
        s = self._section(af, "📌 Contexto del problema", 2)
        self._body(s,
            "En los centros de datos modernos (data centers), los servidores generan grandes cantidades "
            "de calor que deben eliminarse con sistemas de enfriamiento por fluido. El refrigerante circula "
            "en patrones de flujo bidimensional entre los racks de servidores.", 1)
        self._body(s,
            "Las líneas de flujo (streamlines) del refrigerante son curvas que en cada punto son "
            "tangentes a la velocidad del fluido. Para un flujo particular, estas líneas satisfacen "
            "una Ecuación Diferencial Ordinaria homogénea de primer orden.", 2)

        # Modelo matemático
        s2 = self._section(af, "📐 Modelo matemático", 3)
        self._body(s2,
            "Si la velocidad del fluido en el punto (x, y) tiene componentes proporcionales a las "
            "funciones M(x,y) en dirección x y N(x,y) en dirección y, la trayectoria de una "
            "partícula de fluido satisface:", 1)
        ctk.CTkLabel(s2, text="  M(x,y)·dx  +  N(x,y)·dy  =  0",
                     font=ctk.CTkFont("Courier New", 15, "bold"),
                     text_color=GREEN).grid(row=2, column=0, padx=14, pady=6, sticky="w")
        self._body(s2,
            "Para el flujo de enfriamiento entre racks de servidores, se ha determinado experimentalmente "
            "que el campo de velocidades satisface la relación homogénea de grado 2:", 3)
        ctk.CTkLabel(s2, text="  M(x,y) = x² + y²    →    N(x,y) = −2xy",
                     font=ctk.CTkFont("Courier New", 13),
                     text_color=YEL).grid(row=4, column=0, padx=14, pady=4, sticky="w")
        self._body(s2,
            "La solución general de esta ecuación da la familia de curvas que describen las "
            "trayectorias del refrigerante: x² = C·y  (parábolas). Conocer estas trayectorias "
            "permite a los ingenieros de sistemas diseñar la distribución óptima de los ductos.", 5)

        # ED a resolver
        s3 = self._section(af, "✏️  Ecuación a resolver en el solucionador", 4)
        self._body(s3, "Ingresa los siguientes valores en el solucionador:", 1)
        ctk.CTkLabel(s3, text="  M(x,y) = x² + y²        N(x,y) = −2xy",
                     font=ctk.CTkFont("Courier New", 14, "bold"),
                     text_color=CYAN).grid(row=2, column=0, padx=14, pady=6, sticky="w")
        ctk.CTkButton(s3, text="⬅  Cargar este ejemplo en el solucionador",
                       height=38, corner_radius=8,
                       fg_color=ACC, hover_color="#6655ee",
                       font=ctk.CTkFont("Segoe UI", 12),
                       command=self._load_app_t1
                       ).grid(row=3, column=0, padx=14, pady=(4,12), sticky="w")

    def _load_app_t1(self):
        self._fill1(
            [("x**2","x²","var"),("+","+","op"),("y**2","y²","var")],
            [("-","−","_neg"),("2","2","d"),("x","x","var"),("y","y","var")]
        )
        self.tabs.set("📋  Procedimiento paso a paso")

    def _draw_datacenter(self, ax):
        ax.set_xlim(0, 12); ax.set_ylim(0, 5)
        ax.set_facecolor(INP); ax.axis('off')

        # Racks de servidores
        for xi in [0.5, 4.2, 7.9]:
            ax.add_patch(mpatches.FancyBboxPatch((xi, 0.5), 2.8, 4,
                boxstyle="round,pad=0.1", fc="#1a1a35", ec="#4a4a8a", lw=2))
            for yi in np.linspace(0.9, 4.0, 6):
                ax.add_patch(mpatches.FancyBboxPatch((xi+0.2, yi), 2.4, 0.35,
                    boxstyle="round,pad=0.05", fc="#0a2a4a", ec="#2a4a7a", lw=1))
            ax.text(xi+1.4, 0.1, "RACK", ha='center', color=DIM, fontsize=7)

        # Streamlines (trayectorias del refrigerante)
        x_arr = np.linspace(-2, 2, 400)
        for C_val in [-8, -3, -1, 1, 3, 8]:
            if C_val > 0:
                y_arr = x_arr**2 / C_val
                mask = (y_arr > 0) & (y_arr < 5)
                ax.plot(x_arr[mask]*0.7 + 6, y_arr[mask]*0.8 + 0.3,
                         color='#4cc9f0', alpha=0.6, linewidth=1.5, zorder=3)

        ax.annotate("", xy=(6.9, 3.5), xytext=(6.4, 3.2),
                     arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.5))
        ax.text(7.0, 3.8, "Trayectorias\ndel refrigerante\nx²=Cy",
                color=CYAN, fontsize=8, ha='center')
        ax.text(5.8, 0.2, "ED: (x²+y²)dx − 2xy·dy = 0",
                color=YEL, fontsize=9, ha='center', fontweight='bold')
        ax.set_title("Sistema de enfriamiento de data center — líneas de flujo",
                     color=MAIN, fontsize=10, pad=6)

    # ── Aplicación Tipo 2: Vibración HDD en rack ──────────────────────────────
    def _build_app_t2(self):
        af = self.app_frame

        # Título
        ttl = ctk.CTkFrame(af, fg_color="#1a0e2e", corner_radius=12)
        ttl.grid(row=0, column=0, padx=6, pady=(6,4), sticky="ew")
        ttl.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ttl, text="💿  Aplicación: Vibración de disco duro en rack de servidores",
                     font=ctk.CTkFont("Segoe UI", 14, "bold"),
                     text_color=ACC).grid(row=0, column=0, padx=14, pady=(12,4), sticky="w")
        ctk.CTkLabel(ttl,
                     text="ED homogénea de 2° orden para análisis de vibraciones mecánicas en hardware",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=1, column=0, padx=14, pady=(0,12), sticky="w")

        # Diagrama + sliders
        diag_frame = ctk.CTkFrame(af, fg_color=CARD, corner_radius=10)
        diag_frame.grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        diag_frame.grid_columnconfigure(0, weight=1)

        # Diagrama
        self.fig_app2 = Figure(figsize=(7, 2.6), dpi=96)
        self.fig_app2.patch.set_facecolor(CARD)
        self.ax_app2 = self.fig_app2.add_subplot(111)
        self.cv_app2 = FigureCanvasTkAgg(self.fig_app2, master=diag_frame)
        self.cv_app2.get_tk_widget().configure(bg=CARD)
        self.cv_app2.get_tk_widget().grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self._draw_hdd_system()

        # Sliders de parámetros
        sl_frame = ctk.CTkFrame(diag_frame, fg_color="transparent")
        sl_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=(0,10))
        sl_frame.grid_columnconfigure((0,1,2), weight=1)

        params = [
            ("m  (masa kg)",    self._m_val, 0.1, 3.0,  "kg", "mass"),
            ("k  (rigidez N/m)",self._k_val, 10,  300.0, "N/m","spring"),
            ("c  (amort. N·s/m)",self._c_val,0,   30.0, "N·s/m","damp"),
        ]
        self.sl_labels = {}
        for col,(lbl,var,mn,mx,unit,tag) in enumerate(params):
            fr = ctk.CTkFrame(sl_frame, fg_color="transparent")
            fr.grid(row=0, column=col, padx=6)
            ctk.CTkLabel(fr, text=lbl, font=ctk.CTkFont("Segoe UI",10),
                          text_color=MID).pack()
            lv = ctk.CTkLabel(fr, text=f"{var.get():.1f}",
                               font=ctk.CTkFont("Courier New",13,"bold"),
                               text_color=ACC)
            lv.pack()
            self.sl_labels[tag] = lv
            ctk.CTkSlider(fr, from_=mn, to=mx, variable=var,
                           width=120, button_color=ACC,
                           command=lambda v, t=tag, lv=lv, u=unit: self._on_slider(v,t,lv,u)
                           ).pack(pady=4)
            ctk.CTkLabel(fr, text=unit, font=ctk.CTkFont("Segoe UI",9),
                          text_color=DIM).pack()

        # ED resultante
        self.app_eq_lbl = ctk.CTkLabel(diag_frame, text="",
                                        font=ctk.CTkFont("Courier New",14,"bold"),
                                        text_color=GREEN)
        self.app_eq_lbl.grid(row=2, column=0, pady=6)
        self._update_app2_eq()

        ctk.CTkButton(diag_frame,
                       text="⬅  Cargar esta ecuación en el solucionador",
                       height=36, corner_radius=8,
                       fg_color=ACC, hover_color="#6655ee",
                       font=ctk.CTkFont("Segoe UI",12),
                       command=self._load_app_t2
                       ).grid(row=3, column=0, padx=14, pady=(0,12), sticky="w")

        # Contexto
        s = self._section(af, "📌 Contexto del problema", 2)
        self._body(s,
            "Los discos duros (HDD) en servidores son componentes mecánicos sensibles a las vibraciones. "
            "Las vibraciones pueden causar errores de lectura/escritura, reducir la vida útil del disco "
            "o incluso causar fallas catastróficas. Por ello, los racks de servidores utilizan monturas "
            "de aislamiento (goma, resortes, amortiguadores) para proteger los discos.", 1)

        s2 = self._section(af, "📐 Modelo matemático", 3)
        self._body(s2,
            "Aplicando la Segunda Ley de Newton al disco duro sobre su montura elástica-amortiguada, "
            "la suma de fuerzas en vibración libre (sin fuerza externa aplicada) da:", 1)
        ctk.CTkLabel(s2, text="  m·y''(t)  +  c·y'(t)  +  k·y(t)  =  0",
                     font=ctk.CTkFont("Courier New",15,"bold"),
                     text_color=GREEN).grid(row=2,column=0,padx=14,pady=6,sticky="w")
        self._body(s2,
            "Donde:  y(t) = desplazamiento del disco [m]  |  y'(t) = velocidad [m/s]\n"
            "         y''(t) = aceleración [m/s²]  |  m = masa [kg]  |  k = rigidez [N/m]  |  c = amortiguamiento [N·s/m]", 3)
        self._body(s2,
            "Esta es exactamente una ED homogénea lineal de 2° orden con coeficientes constantes. "
            "Sus 3 casos (discriminante positivo, cero o negativo) corresponden físicamente a:\n"
            "• Δ > 0: Sistema sobre-amortiguado (el disco regresa lentamente, sin oscilar)\n"
            "• Δ = 0: Amortiguamiento crítico (retorno más rápido posible sin oscilación)\n"
            "• Δ < 0: Sub-amortiguado (el disco oscila con amplitud decreciente — lo más común)", 4)

    def _on_slider(self, val, tag, lv, unit):
        lv.configure(text=f"{float(val):.1f}")
        self._draw_hdd_system()
        self._update_app2_eq()

    def _update_app2_eq(self):
        m = self._m_val.get(); k = self._k_val.get(); c = self._c_val.get()
        m_s = f"{m:.1f}"; k_s = f"{k:.0f}"; c_s = f"{c:.1f}"
        eq = f"  {m_s}y'' + {c_s}y' + {k_s}y = 0"
        disc = c**2 - 4*m*k
        if   disc > 0:  caso = "Δ > 0  →  Sobre-amortiguado"
        elif disc == 0: caso = "Δ = 0  →  Crítico"
        else:           caso = f"Δ = {disc:.1f}  →  Sub-amortiguado (oscila)"
        if hasattr(self, 'app_eq_lbl'):
            self.app_eq_lbl.configure(text=f"{eq}\n  {caso}")

    def _load_app_t2(self):
        m = round(self._m_val.get(), 1)
        k = round(self._k_val.get())
        c = round(self._c_val.get(), 1)
        for key, val in [("a",m),("b",c),("c",k)]:
            toks = []
            sv = str(val)
            if val < 0:
                toks.append(("-","−","_neg"))
                toks.append((str(abs(val)),str(abs(val)),"d"))
            elif val != 0:
                toks.append((sv,sv,"d"))
            self.t2f[key].set_tokens(toks)
        self.tabs.set("📋  Procedimiento paso a paso")

    def _draw_hdd_system(self):
        ax = self.ax_app2
        ax.clear()
        m = self._m_val.get(); k = self._k_val.get(); c = self._c_val.get()
        ax.set_xlim(0, 10); ax.set_ylim(0, 5)
        ax.set_facecolor(CARD); ax.axis('off')

        # Techo (rack)
        ax.fill_between([1, 9], [4.3, 4.3], [4.6, 4.6], color='#4a4a8a')
        ax.text(5, 4.8, "RACK de servidor", ha='center', color=MID, fontsize=9)

        # Resorte (izquierda)
        sx = 3.0; sy_top = 4.3; sy_bot = 2.2
        n_coils = 8
        sx_vals = [sx]; sy_vals = [sy_top]
        for i in range(n_coils*2):
            sx_vals.append(sx + 0.25*((-1)**i))
            sy_vals.append(sy_top - (i+1)*(sy_top-sy_bot)/(n_coils*2))
        sx_vals.append(sx); sy_vals.append(sy_bot)
        ax.plot(sx_vals, sy_vals, color=CYAN, lw=2)
        ax.text(2.4, 3.2, f"k={k:.0f}\nN/m", color=CYAN, fontsize=8, ha='center')

        # Amortiguador (derecha)
        dx = 7.0; dy_top = 4.3; dy_mid = 3.3; dy_bot = 2.2
        ax.plot([dx, dx], [dy_top, dy_mid], color=YEL, lw=2)
        ax.add_patch(mpatches.FancyBboxPatch((dx-0.3, dy_mid-0.3), 0.6, 0.6,
            boxstyle="square", fc="#1a1a35", ec=YEL, lw=2))
        ax.plot([dx-0.25, dx-0.25], [dy_mid-0.3, dy_bot], color=YEL, lw=2)
        ax.plot([dx+0.25, dx+0.25], [dy_mid-0.3, dy_bot], color=YEL, lw=2)
        ax.plot([dx-0.3, dx+0.3], [dy_bot, dy_bot], color=YEL, lw=2)
        ax.text(7.8, 3.2, f"c={c:.1f}\nN·s/m", color=YEL, fontsize=8, ha='center')

        # Disco duro (masa)
        ax.add_patch(mpatches.FancyBboxPatch((3.5, 1.4), 3, 0.8,
            boxstyle="round,pad=0.1", fc="#2a2a6a", ec=ACC, lw=2))
        ax.text(5, 1.8, f"💿 HDD   m = {m:.1f} kg", ha='center', color=MAIN, fontsize=10, fontweight='bold')

        # Flecha de desplazamiento
        ax.annotate("", xy=(9.2, 1.5), xytext=(9.2, 2.5),
                     arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
        ax.text(9.5, 2.0, "y(t)", color=RED, fontsize=10, va='center')

        ax.set_title(f"Sistema masa-resorte-amortiguador  |  ED: {m:.1f}y'' + {c:.1f}y' + {k:.0f}y = 0",
                     color=MAIN, fontsize=9, pad=4)
        if hasattr(self, 'cv_app2'):
            self.cv_app2.draw()

    # ══════════════════════════════════════════════════════════════════════════
    #  SOLVER PRINCIPAL
    # ══════════════════════════════════════════════════════════════════════════
    def _solve(self):
        self.btn_s.configure(state="disabled", text="⏳  Calculando...")
        self.update()
        try:
            if self.eq_type.get() == "tipo1": self._t1()
            else:                               self._t2()
        except Exception as e:
            self._err(f"Error inesperado:\n  {e}")
        finally:
            self.btn_s.configure(state="normal", text="▶   Resolver paso a paso")

    # ── TIPO 1 ────────────────────────────────────────────────────────────────
    def _t1(self):
        me = self.fm.get_expr().strip()
        ne = self.fn.get_expr().strip()
        md = self.fm.get_disp().strip()
        nd = self.fn.get_disp().strip()
        if not me or not ne:
            self._err("Ingresa M(x,y) y N(x,y) usando el teclado."); return
        try:
            M = sp.sympify(me, locals={"x":xs,"y":ys})
            N = sp.sympify(ne, locals={"x":xs,"y":ys})
        except Exception as e:
            self._err(f"Expresión no válida:\n  {e}"); return

        EQ="═"*56; SP="─"*56; L=[]
        L+=[EQ,"   ECUACIÓN HOMOGÉNEA DE PRIMER ORDEN",EQ,"",
            f"   ({md})·dx  +  ({nd})·dy  =  0",""]

        # Paso 1
        L+=[SP,"   PASO 1: Verificar homogeneidad de M y N",SP,""]
        Mt=sp.expand(M.subs([(xs,ts*xs),(ys,ts*ys)]))
        Nt=sp.expand(N.subs([(xs,ts*xs),(ys,ts*ys)]))
        g=None
        for n in range(7):
            try:
                Ms=sp.simplify(Mt/ts**n); Ns=sp.simplify(Nt/ts**n)
                if not Ms.has(ts) and not Ns.has(ts): g=n; break
            except: continue
        if g is None:
            L+=["   ✗ La ecuación NO parece ser homogénea.",
                "   Verifica que M y N tengan el mismo grado."]
            self._wrt(L); return
        L+=[f"   M(tx,ty) = t^{g} · M(x,y)  ✓",
            f"   N(tx,ty) = t^{g} · N(x,y)  ✓","",
            f"   → La ecuación es homogénea de grado {g}  ✓",""]

        # Paso 2
        L+=[SP,"   PASO 2: Sustitución  y = vx",SP,"",
            "   Sea:    y  = v · x          (v = v(x))",
            "           dy = v·dx + x·dv",""]

        # Paso 3
        L+=[SP,"   PASO 3: Sustituir en la ecuación",SP,""]
        Mv=sp.simplify(M.subs(ys,vs*xs)); Nv=sp.simplify(N.subs(ys,vs*xs))
        tot=sp.simplify(Mv+vs*Nv)
        L+=[f"   M(x, vx) = {Mv}",f"   N(x, vx) = {Nv}","",
            "   La ecuación se convierte en:",
            f"   ({sp.factor(Mv)})·dx  +  ({sp.factor(Nv)})·(v·dx + x·dv) = 0","",
            "   Expandiendo y agrupando en dx y en x·dv:",
            f"   [{sp.factor(tot)}]·dx  +  [{sp.factor(Nv)}]·x·dv = 0",""]

        # Paso 4
        L+=[SP,"   PASO 4: Separar variables (usando x=1 para factorizar)",SP,""]
        M1=sp.simplify(M.subs([(xs,1),(ys,vs)]))
        N1=sp.simplify(N.subs([(xs,1),(ys,vs)]))
        A=sp.simplify(M1+vs*N1); B=sp.simplify(N1)
        if A==0:
            L+=["   ✗ A(v) = 0 — revisar la ecuación."]; self._wrt(L); return
        ig=sp.simplify(-B/A)
        L+=[f"   M(1,v) = {M1}  →  Coef. de dx: A(v) = {A}",
            f"   N(1,v) = {N1}  →  Coef. de dv: B(v) = {B}","",
            f"   Factorizando x^{g} y dividiendo entre x^{g+1}:","",
            "   Resultado separado:",f"   dx/x  =  ({ig})·dv",""]

        # Paso 5
        L+=[SP,"   PASO 5: Integrar ambos lados",SP,"",
            f"   ∫ dx/x  =  ∫ ({ig}) dv",""]
        try:
            itg=sp.simplify(sp.integrate(ig,vs))
            L+=[f"   ln|x|  =  {itg}  +  C",""]
        except:
            L+=["   (La integral no pudo resolverse simbólicamente —","    resuélvela manualmente.)",""]
            self._wrt(L); return

        # Paso 6
        L+=[SP,"   PASO 6: Retrosustitución  v = y/x",SP,""]
        try:
            sol=sp.simplify(itg.subs(vs,ys/xs))
            L+=["   ╔══════════════════════════════════════════╗",
                "   ║   ✅  SOLUCIÓN GENERAL                  ║",
                "   ╚══════════════════════════════════════════╝","",
                f"   ln|x|  =  {sol}  +  C","",
                "   C = constante arbitraria (determinada por","   condición inicial si se dispone de ella)",""]
        except:
            L+=[f"   Sustituye v = y/x en:   {itg}  +  C",""]
        L+=[EQ]
        self._wrt(L); self._plt1(M,N)

    # ── TIPO 2 ────────────────────────────────────────────────────────────────
    def _t2(self):
        try:
            af=float(self.t2f["a"].get_expr() or "0")
            bf=float(self.t2f["b"].get_expr() or "0")
            cf=float(self.t2f["c"].get_expr() or "0")
        except ValueError:
            self._err("Ingresa valores numéricos en a, b y c\n(haz clic en el campo y usa el teclado)."); return
        if af==0:
            self._err("El coeficiente 'a' no puede ser cero."); return

        a=sp.Rational(af).limit_denominator(1000)
        b=sp.Rational(bf).limit_denominator(1000)
        c=sp.Rational(cf).limit_denominator(1000)

        EQ="═"*56; SP="─"*56; L=[]
        L+=[EQ,"   ECUACIÓN HOMOGÉNEA LINEAL DE SEGUNDO ORDEN",EQ,"",
            f"   {self._feq(af,bf,cf)}  =  0",""]

        L+=[SP,"   PASO 1: Identificar coeficientes",SP,"",
            f"   a = {a}   →  coeficiente de y''",
            f"   b = {b}   →  coeficiente de y'",
            f"   c = {c}   →  coeficiente de y",""]

        L+=[SP,"   PASO 2: Ecuación característica",SP,"",
            "   Proponemos la solución de forma:  y = e^(rx)",
            "   Derivando:  y' = r·e^(rx)   y   y'' = r²·e^(rx)","",
            "   Sustituyendo en la ED y factorizando e^(rx) ≠ 0:",
            f"   {self._fch(af,bf,cf)} = 0",""]

        disc=b**2-4*a*c; df=float(disc)
        L+=[SP,"   PASO 3: Calcular el discriminante  Δ = b² − 4ac",SP,"",
            f"   Δ = ({b})² − 4·({a})·({c})",
            f"   Δ = {b**2} − {4*a*c}",
            f"   Δ = {disc}",""]

        r1s=sp.simplify((-b+sp.sqrt(disc))/(2*a))
        r2s=sp.simplify((-b-sp.sqrt(disc))/(2*a))
        L+=[SP,"   PASO 4: Calcular las raíces",SP,"",
            "   Fórmula cuadrática:  r = (−b ± √Δ) / 2a",
            f"   r = (−({b}) ± √({disc})) / (2·{a})","",
            f"   r₁ = {r1s}",f"   r₂ = {r2s}",""]

        L+=[SP,"   PASO 5: Determinar el caso según Δ",SP,""]
        if df>0:
            cs="dist"; r1n,r2n=float(r1s),float(r2s); alp=bet=None
            L+=[f"   Δ = {disc} > 0",
                "   → CASO 1: Raíces reales y distintas","",
                "     Interpretación física (sistema HDD):",
                "     El disco regresa a equilibrio sin oscilar",
                "     (sistema SOBRE-amortiguado)","",
                f"   r₁ = {r1s}",f"   r₂ = {r2s}",""]
        elif df==0:
            cs="rep"; r1n=r2n=float(r1s); alp=bet=None
            L+=["   Δ = 0",
                "   → CASO 2: Raíz real repetida","",
                "     Interpretación física (sistema HDD):",
                "     AMORTIGUAMIENTO CRÍTICO — el retorno más",
                "     rápido posible sin oscilar","",
                f"   r₁ = r₂ = {r1s}",""]
        else:
            cs="comp"
            als=sp.simplify(-b/(2*a)); bes=sp.simplify(sp.sqrt(-disc)/(2*a))
            alp,bet=float(als),float(bes)
            r1n=complex(alp,bet); r2n=complex(alp,-bet)
            L+=[f"   Δ = {disc} < 0",
                "   → CASO 3: Raíces complejas conjugadas","",
                "     Interpretación física (sistema HDD):",
                "     El disco OSCILA con amplitud que decrece",
                "     exponencialmente (sub-amortiguado)","",
                "   Forma:  r = α ± βi",
                f"   α (parte real)        = {als}   ← decaimiento",
                f"   β (parte imaginaria)  = {bes}   ← frecuencia","",
                f"   r₁ = {als} + {bes}i",f"   r₂ = {als} − {bes}i",""]

        L+=[SP,"   PASO 6: Solución general",SP,""]
        if cs=="dist":
            sol=f"   y(t) = C₁·e^({r1s}t)  +  C₂·e^({r2s}t)"
        elif cs=="rep":
            sol=f"   y(t) = (C₁  +  C₂·t)·e^({r1s}t)"
        else:
            als2=sp.simplify(-b/(2*a)); bes2=sp.simplify(sp.sqrt(-disc)/(2*a))
            pr=f"e^({als2}t)·" if alp!=0 else ""
            sol=f"   y(t) = {pr}[C₁·cos({bes2}t)  +  C₂·sin({bes2}t)]"

        L+=["   ╔══════════════════════════════════════════╗",
            "   ║   ✅  SOLUCIÓN GENERAL                  ║",
            "   ╚══════════════════════════════════════════╝",
            "",sol,"",
            "   C₁, C₂ = constantes determinadas por condiciones",
            "   iniciales: posición y velocidad en t = 0","",EQ]

        self._wrt(L); self._plt2(cs,r1n,r2n,alp,bet)

    # ══════════════════════════════════════════════════════════════════════════
    #  GRÁFICAS
    # ══════════════════════════════════════════════════════════════════════════
    def _sax(self, ax, title):
        ax.set_facecolor("#0a0a1e")
        ax.set_title(title, color=MAIN, fontsize=11, pad=10, fontweight="bold")
        ax.set_xlabel("x", color=MID, fontsize=11)
        ax.set_ylabel("y", color=MID, fontsize=11)
        ax.tick_params(colors=DIM)
        ax.axhline(0, color=SEP, lw=0.8, zorder=0)
        ax.axvline(0, color=SEP, lw=0.8, zorder=0)
        for s in ax.spines.values(): s.set_color(SEP)

    def _blank_graph(self):
        self.fig_g.clear(); self.fig_g.patch.set_facecolor(INP)
        ax=self.fig_g.add_subplot(111); ax.set_facecolor("#0a0a1e")
        ax.text(0.5,0.5,"Resuelve una ecuación para ver la gráfica aquí",
                ha="center",va="center",color=DIM,fontsize=13,transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(SEP)
        self.cv_g.draw()

    def _plt1(self, M, N):
        self.fig_g.clear(); self.fig_g.patch.set_facecolor(INP)
        ax=self.fig_g.add_subplot(111)
        self._sax(ax,"Campo de direcciones  —  dy/dx = −M(x,y) / N(x,y)")
        xv=np.linspace(-4,4,26); yv=np.linspace(-4,4,26)
        Xg,Yg=np.meshgrid(xv,yv)
        try:
            Mf=sp.lambdify((xs,ys),M,"numpy")
            Nf=sp.lambdify((xs,ys),N,"numpy")
            with np.errstate(all="ignore"):
                Mv=np.array(Mf(Xg,Yg),dtype=float)
                Nv=np.array(Nf(Xg,Yg),dtype=float)
                U=np.where(np.isfinite(Nv),Nv,0.)
                V=np.where(np.isfinite(Mv),-Mv,0.)
                nm=np.sqrt(U**2+V**2); nm[nm==0]=1
                U/=nm; V/=nm
                mask=np.isfinite(U)&np.isfinite(V)
            ax.quiver(Xg[mask],Yg[mask],U[mask],V[mask],
                       color=CYAN, alpha=0.72, scale=28,
                       width=0.003, headwidth=4, headlength=5)
            ax.set_xlim(-4,4); ax.set_ylim(-4,4)
        except Exception as e:
            ax.text(0.5,0.5,f"No se pudo graficar:\n{e}",
                     ha="center",va="center",color="#a05050",transform=ax.transAxes)
        self.fig_g.tight_layout(pad=1.8); self.cv_g.draw()
        self.tabs.set("📈  Gráfica de solución")

    def _plt2(self, cs, r1, r2, alp, bet):
        self.fig_g.clear(); self.fig_g.patch.set_facecolor(INP)
        ax=self.fig_g.add_subplot(111)
        tt={"dist":"Sobre-amortiguado — Raíces reales distintas",
            "rep" :"Amortiguamiento crítico — Raíz repetida",
            "comp":"Sub-amortiguado — Raíces complejas (oscila)"}
        self._sax(ax,f"Posición y(t) del HDD  —  {tt[cs]}")
        ax.set_xlabel("t  (tiempo)", color=MID, fontsize=11)
        ax.set_ylabel("y(t)  (desplazamiento)", color=MID, fontsize=11)
        xp=np.linspace(0,8,800)
        ics=[(1,0),(0,1),(1,0.5),(1,-0.5),(0.5,0.5)]
        for i,(c1,c2) in enumerate(ics):
            with np.errstate(all="ignore"):
                if cs=="dist":
                    r1r=r1.real if isinstance(r1,complex) else float(r1)
                    r2r=r2.real if isinstance(r2,complex) else float(r2)
                    yp=c1*np.exp(r1r*xp)+c2*np.exp(r2r*xp)
                elif cs=="rep":
                    r1r=r1.real if isinstance(r1,complex) else float(r1)
                    yp=(c1+c2*xp)*np.exp(r1r*xp)
                else:
                    yp=np.exp(alp*xp)*(c1*np.cos(bet*xp)+c2*np.sin(bet*xp))
            mask=np.isfinite(yp)&(np.abs(yp)<15)
            if mask.any():
                ax.plot(xp[mask],yp[mask],color=PAL[i],lw=1.8,alpha=.9,
                         label=f"y(0)={c1}, y'(0)={c2}")
        ax.set_xlim(0,8); ax.set_ylim(-6,6)
        ax.legend(fontsize=8, facecolor="#12122a", labelcolor=MAIN,
                   edgecolor=SEP, loc="best", framealpha=.85)
        self.fig_g.tight_layout(pad=1.8); self.cv_g.draw()
        self.tabs.set("📈  Gráfica de solución")

    # ══════════════════════════════════════════════════════════════════════════
    #  UTILIDADES
    # ══════════════════════════════════════════════════════════════════════════
    def _feq(self, a, b, c):
        parts=[]
        for val,term in [(a,"y''"),(b,"y'"),(c,"y")]:
            if val==0: continue
            if not parts:
                if val==1:   parts.append(term)
                elif val==-1:parts.append(f"−{term}")
                else:        parts.append(f"{val:g}{term}")
            else:
                if val==1:   parts.append(f"+ {term}")
                elif val==-1:parts.append(f"− {term}")
                elif val>0:  parts.append(f"+ {val:g}{term}")
                else:        parts.append(f"− {abs(val):g}{term}")
        return " ".join(parts) or "0"

    def _fch(self, a, b, c):
        parts=[]
        for val,term in [(a,"r²"),(b,"r"),(c,"")]:
            if val==0: continue
            d=f"{val:g}{term}" if term else f"{val:g}"
            if not parts: parts.append(d)
            elif val>0:   parts.append(f"+ {d}")
            else:         parts.append(f"− {abs(val):g}{term}")
        return " ".join(parts)

    def _wrt(self, lines):
        self.out.configure(state="normal")
        self.out.delete("1.0","end")
        for l in lines: self.out.insert("end",l+"\n")
        self.out.see("1.0"); self.out.configure(state="disabled")
        self.tabs.set("📋  Procedimiento paso a paso")
        self.stlbl.configure(text="✓ Solución generada correctamente",text_color=GREEN)

    def _err(self, msg):
        self._wrt(["","   ⚠  ERROR","",f"   {msg}",""])
        self.stlbl.configure(text="Error — revisa los datos",text_color=RED)

    def _clear(self):
        self._welcome()
        self._blank_graph()
        self.stlbl.configure(text="")
        if hasattr(self,"fm"):
            self.fm.clear(); self.fn.clear()
        if hasattr(self,"t2f"):
            for fld in self.t2f.values(): fld.clear()

    def _welcome(self):
        EQ="═"*56; SP="─"*56
        lines=[EQ,"   Bienvenido al Solucionador de EDs Homogéneas",EQ,"",
            "   Instrucciones:","",
            "   1.  Elige el tipo de ecuación en el panel izquierdo",
            "   2.  Haz clic en 🔬 Problema para ver el contexto real",
            "   3.  Haz clic en el campo a llenar y usa el teclado",
            "   4.  Presiona  ▶ Resolver paso a paso",
            "   5.  Lee el procedimiento | ve la gráfica","",SP,
            "   TIPO 1 — Primer orden  M(x,y)dx + N(x,y)dy = 0",
            "   Aplicación: Líneas de flujo en enfriamiento de Data Center",
            "   • Verificación de homogeneidad",
            "   • Sustitución y = vx",
            "   • Separación de variables e integración",
            "   • Retrosustitución v = y/x","",
            "   TIPO 2 — Segundo orden  ay'' + by' + cy = 0",
            "   Aplicación: Vibración de disco duro en rack de servidor",
            "   • Ecuación característica  ar² + br + c = 0",
            "   • Caso 1 (Δ>0): raíces reales → sobre-amortiguado",
            "   • Caso 2 (Δ=0): raíz repetida → amortiguamiento crítico",
            "   • Caso 3 (Δ<0): raíces complejas → sub-amortiguado",SP]
        self.out.configure(state="normal")
        self.out.delete("1.0","end")
        for l in lines: self.out.insert("end",l+"\n")
        self.out.see("1.0"); self.out.configure(state="disabled")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()