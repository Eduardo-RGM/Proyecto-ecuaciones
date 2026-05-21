"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SOLUCIONADOR DE ECUACIONES DIFERENCIALES HOMOGÉNEAS                       ║
║   Proyecto Integrador — Ingeniería en Sistemas Computacionales               ║
║   Versión 5.0 — Correcciones completas                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Tipo 1 (1° Orden):  M(x,y)dx + N(x,y)dy = 0  →  Enfriamiento de CPUs     ║
║   Tipo 2 (2° Orden):  ay″ + by′ + cy = 0        →  Circuito RLC             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Instalación:  pip install customtkinter matplotlib sympy numpy             ║
║   Ejecución:    python ed_homogeneas_v5.py                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

CORRECCIONES v5:
  ✔ set_tokens() ahora inserta '*' implícitos → expresiones como -2xy se
    convierten correctamente en -2*x*y antes de parsear con SymPy.
  ✔ El foco del campo activo funciona correctamente: clic en cualquier parte
    de la tarjeta (label, borde, cursor) activa ese campo.
  ✔ Al resolver, se navega primero a "Solución Paso a Paso", no a la gráfica.
  ✔ Los ejemplos predefinidos cargan tokens con '*' correctos.
  ✔ Entrada por TECLADO FÍSICO habilitada para tipo 2 (solo dígitos y signo).
  ✔ Mensajes de ayuda claros sobre el formato de entrada esperado.
"""
import warnings; warnings.filterwarnings("ignore")
import customtkinter as ctk
import sympy as sp
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

# ─── Símbolos SymPy ──────────────────────────────────────────────────────────
xs = sp.Symbol("x", real=True)
ys = sp.Symbol("y", real=True)
vs = sp.Symbol("v", real=True)
ts = sp.Symbol("t", positive=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Paleta de colores ────────────────────────────────────────────────────────
BG     = "#0d1117"
SURF   = "#161b22"
CARD   = "#21262d"
INP    = "#0d1117"
BDR    = "#30363d"
BACT   = "#388bfd"
TEXT   = "#e6edf3"
DIM    = "#8b949e"
ACC    = "#388bfd"
GREEN  = "#3fb950"
YELLOW = "#e3b341"
RED    = "#f85149"
PURPLE = "#bc8cff"
CYAN   = "#79c0ff"

PAL = ["#388bfd", "#3fb950", "#ffa657", "#bc8cff", "#79c0ff", "#f85149"]

# Variables para el teclado de Tipo 1
VARS_T1 = [
    ("x",   "x",         "x",    "var"),
    ("y",   "y",         "y",    "var"),
    ("x²",  "x**2",      "x²",   "var"),
    ("y²",  "y**2",      "y²",   "var"),
    ("xy",  "x*y",       "xy",   "var"),
    ("x³",  "x**3",      "x³",   "var"),
    ("x²y", "x**2*y",    "x²y",  "var"),
    ("xy²", "x*y**2",    "xy²",  "var"),
    ("y³",  "y**3",      "y³",   "var"),
    ("(",   "(",          "(",    "pop"),
    (")",   ")",          ")",    "pcl"),
    ("/",   "/",          "÷",    "op"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDAD: construir lista de tokens CON multiplicaciones implícitas
# ═══════════════════════════════════════════════════════════════════════════════
def _needs_mul(prev_kind: str, next_kind: str) -> bool:
    """¿Se necesita insertar '*' entre dos tokens consecutivos?"""
    return (
        (next_kind in ("var", "pop")) and prev_kind in ("d", "var", "pcl") or
        (next_kind == "d")            and prev_kind in ("var", "pcl")
    )

def build_tokens(raw: list) -> list:
    """
    Recibe una lista de tuplas (internal, display, kind) SIN tokens de
    multiplicación y devuelve la lista completa con los '*' insertados donde
    corresponde, igual que lo hace ExprField.push() en tiempo real.
    Fusiona además dígitos consecutivos.
    """
    result = []
    for tok in raw:
        internal, display, kind = tok
        if result:
            lk = result[-1][2]
            # Fusión de dígitos
            if kind == "d" and lk == "d":
                result[-1] = (result[-1][0] + internal,
                              result[-1][1] + display, "d")
                continue
            if _needs_mul(lk, kind):
                result.append(("*", "·", "_m"))
        result.append((internal, display, kind))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#   WIDGET: Campo de expresión con tokens
# ═══════════════════════════════════════════════════════════════════════════════
class ExprField(ctk.CTkFrame):
    """
    Campo de entrada matemática basado en tokens.
    CORRECCIÓN: bind en TODOS los widgets hijos para capturar clics.
    """
    def __init__(self, master, label="", placeholder="Clic aquí → usa el teclado", **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self._tok = []
        self._ph  = placeholder

        if label:
            self._lbl_title = ctk.CTkLabel(
                self, text=label,
                font=ctk.CTkFont("Consolas", 13, "bold"),
                text_color=CYAN)
            self._lbl_title.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 2))

        self._box = ctk.CTkFrame(self, fg_color=INP, corner_radius=8,
                                  border_width=2, border_color=BDR)
        self._box.grid(row=1, column=0, sticky="ew")
        self._box.grid_columnconfigure(0, weight=1)

        self._lbl = ctk.CTkLabel(
            self._box, text="",
            font=ctk.CTkFont("Consolas", 15),
            text_color=TEXT, anchor="w")
        self._lbl.grid(row=0, column=0, sticky="ew", padx=12, pady=11)

        self._cur = ctk.CTkLabel(
            self._box, text="",
            font=ctk.CTkFont("Consolas", 18, "bold"),
            text_color=ACC)
        self._cur.grid(row=0, column=1, padx=(0, 10))

        # CORRECCIÓN: bind en TODOS los sub-widgets para que el clic funcione
        for w in (self._box, self._lbl, self._cur, self):
            w.bind("<Button-1>", self._on_click)
        if hasattr(self, "_lbl_title"):
            self._lbl_title.bind("<Button-1>", self._on_click)

        self._refresh()

    def _on_click(self, event):
        self.event_generate("<<FieldClick>>")

    def activate(self, on: bool):
        self._box.configure(border_color=BACT if on else BDR)
        self._cur.configure(text="│" if on else "")

    def push(self, internal: str, display: str, kind: str):
        t = self._tok
        if t:
            lk = t[-1][2]
            if kind == "d" and lk == "d":
                t[-1] = (t[-1][0] + internal, t[-1][1] + display, "d")
                self._refresh(); return
            if _needs_mul(lk, kind):
                t.append(("*", "·", "_m"))
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
        self._tok.clear(); self._refresh()

    def negate(self):
        if self._tok and self._tok[0][2] == "_neg":
            self._tok.pop(0)
        else:
            self._tok.insert(0, ("-", "−", "_neg"))
        self._refresh()

    def set_tokens(self, raw_list: list):
        """
        CORRECCIÓN PRINCIPAL: llama a build_tokens() para insertar '*'
        implícitos antes de asignar, igual que si el usuario tecleara uno a uno.
        """
        self._tok = build_tokens(raw_list)
        self._refresh()

    def get_expr(self) -> str:
        return "".join(t[0] for t in self._tok)

    def get_disp(self) -> str:
        return "".join(t[1] for t in self._tok)

    def _refresh(self):
        d = self.get_disp()
        self._lbl.configure(
            text=d if d else self._ph,
            text_color=TEXT if d else DIM)


# ═══════════════════════════════════════════════════════════════════════════════
#   WIDGET: Previsualización de ecuación
# ═══════════════════════════════════════════════════════════════════════════════
class EqPreview(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=CARD, corner_radius=10,
                          border_width=1, border_color=BDR, **kw)
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="  Vista previa de la ecuación",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=DIM).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")
        self._lbl = ctk.CTkLabel(
            self, text=" — ",
            font=ctk.CTkFont("Consolas", 14),
            text_color=GREEN, wraplength=440, justify="left")
        self._lbl.grid(row=1, column=0, padx=12, pady=(2, 10), sticky="ew")

    def set_t1(self, m_disp: str, n_disp: str):
        m = m_disp if m_disp else "M(x,y)"
        n = n_disp if n_disp else "N(x,y)"
        self._lbl.configure(text=f"  ( {m} ) dx  +  ( {n} ) dy  =  0")

    def set_t2(self, a: str, b: str, c: str):
        parts = []
        for v, t in [(a, "y″"), (b, "y′"), (c, "y")]:
            if not v:
                continue
            if not parts:
                parts.append(f"{v}{t}")
            elif v.startswith("−"):
                parts.append(f"  −  {v[1:]}{t}")
            else:
                parts.append(f"  +  {v}{t}")
        eq = "".join(parts) if parts else "0"
        self._lbl.configure(text=f"  {eq}  =  0")


# ═══════════════════════════════════════════════════════════════════════════════
#   APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ED Homogéneas — Ingeniería en Sistemas Computacionales")
        self.geometry("1460x900")
        self.minsize(1220, 800)
        self.configure(fg_color=BG)
        self._active_field = None
        self._eq_type = "tipo2"
        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  LAYOUT RAÍZ
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        lo = ctk.CTkFrame(self, width=510, corner_radius=0, fg_color=SURF)
        lo.grid(row=0, column=0, sticky="nsew")
        lo.grid_propagate(False)
        lo.grid_rowconfigure(0, weight=1)
        lo.grid_columnconfigure(0, weight=1)

        self._lscroll = ctk.CTkScrollableFrame(
            lo, fg_color="transparent",
            scrollbar_button_color=BDR,
            scrollbar_button_hover_color=ACC)
        self._lscroll.grid(row=0, column=0, sticky="nsew")
        self._lscroll.grid_columnconfigure(0, weight=1)
        self._build_left(self._lscroll)

        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=22, pady=22)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        self._build_right(right)

    # ─────────────────────────────────────────────────────────────────────────
    #  PANEL IZQUIERDO
    # ─────────────────────────────────────────────────────────────────────────
    def _build_left(self, p):
        r = 0

        hdr = ctk.CTkFrame(p, fg_color=CARD, corner_radius=12,
                            border_width=1, border_color=BDR)
        hdr.grid(row=r, column=0, padx=16, pady=(18, 10), sticky="ew"); r += 1
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="ED Homogéneas",
                     font=ctk.CTkFont("Segoe UI", 22, "bold"),
                     text_color=TEXT).grid(row=0, column=0, padx=16, pady=(14, 2), sticky="w")
        ctk.CTkLabel(hdr, text="Ingeniería en Sistemas Computacionales — Proyecto Integrador",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=DIM).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

        ctk.CTkLabel(p, text="  Tipo de ecuación diferencial",
                     font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color=DIM).grid(row=r, column=0, padx=16, pady=(4, 4), sticky="w"); r += 1

        seg = ctk.CTkSegmentedButton(
            p,
            values=["1° Orden", "2° Orden"],
            command=self._on_type_change,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            fg_color=CARD,
            selected_color=ACC, selected_hover_color="#2879f0",
            unselected_color=CARD, unselected_hover_color=BDR,
            text_color=TEXT, text_color_disabled=DIM,
            corner_radius=8, height=42)
        seg.set("2° Orden")
        seg.grid(row=r, column=0, padx=16, pady=(0, 8), sticky="ew"); r += 1

        self._type_sub = ctk.CTkLabel(
            p, text="  ay″ + by′ + cy = 0",
            font=ctk.CTkFont("Consolas", 11),
            text_color=CYAN)
        self._type_sub.grid(row=r, column=0, padx=16, pady=(0, 4), sticky="w"); r += 1

        self._hsep(p, r); r += 1

        self._inp = ctk.CTkFrame(p, fg_color="transparent")
        self._inp.grid(row=r, column=0, padx=16, pady=4, sticky="ew"); r += 1
        self._inp.grid_columnconfigure(0, weight=1)

        self._hsep(p, r); r += 1

        self._prev = EqPreview(p)
        self._prev.grid(row=r, column=0, padx=16, pady=(0, 6), sticky="ew"); r += 1

        self._hsep(p, r); r += 1

        ctk.CTkLabel(p, text="  Teclado matemático",
                     font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color=DIM).grid(row=r, column=0, padx=16, pady=(4, 4), sticky="w"); r += 1

        self._kbd = ctk.CTkFrame(p, fg_color=CARD, corner_radius=12,
                                  border_width=1, border_color=BDR)
        self._kbd.grid(row=r, column=0, padx=16, pady=(0, 6), sticky="ew"); r += 1
        self._kbd.grid_columnconfigure(0, weight=1)

        self._hsep(p, r); r += 1

        self._btn_solve = ctk.CTkButton(
            p, text="▶  Resolver paso a paso",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            height=52, corner_radius=10,
            fg_color=ACC, hover_color="#2879f0",
            command=self._solve)
        self._btn_solve.grid(row=r, column=0, padx=16, pady=(8, 4), sticky="ew"); r += 1

        ctk.CTkButton(
            p, text="Limpiar todo",
            font=ctk.CTkFont("Segoe UI", 11),
            height=36, corner_radius=8,
            fg_color="transparent", border_width=1,
            border_color=BDR, hover_color=CARD,
            text_color=DIM, command=self._clear_all
        ).grid(row=r, column=0, padx=16, pady=(0, 6), sticky="ew"); r += 1

        self._status = ctk.CTkLabel(
            p, text="",
            font=ctk.CTkFont("Segoe UI", 10),
            text_color=DIM, wraplength=474)
        self._status.grid(row=r, column=0, padx=16, pady=(0, 20), sticky="w"); r += 1

        self._build_t2_inputs()
        self._build_keyboard()

    def _hsep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color=BDR).grid(
            row=row, column=0, padx=16, pady=3, sticky="ew")

    # ─────────────────────────────────────────────────────────────────────────
    #  INPUTS — TIPO 1
    # ─────────────────────────────────────────────────────────────────────────
    def _build_t1_inputs(self):
        self._clear_inp_area()
        f = self._inp

        ctk.CTkLabel(f, text="  M(x, y) · dx  +  N(x, y) · dy  =  0",
                     font=ctk.CTkFont("Consolas", 12, "bold"),
                     text_color=CYAN).grid(row=0, column=0, sticky="w", pady=(4, 2))

        # Instrucciones claras
        instr = ctk.CTkFrame(f, fg_color="#0d2a1e", corner_radius=8,
                              border_width=1, border_color="#1a4a2e")
        instr.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        instr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(instr,
                     text="  ① Haz clic en el campo M(x,y) o N(x,y)\n"
                          "  ② Usa el teclado de abajo para ingresar la expresión\n"
                          "  ③ El campo activo se resalta en azul",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=GREEN,
                     justify="left"
                     ).grid(row=0, column=0, padx=8, pady=6, sticky="w")

        self._fm = ExprField(f, label="  M(x, y)  →  coeficiente de dx")
        self._fm.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self._fm.bind("<<FieldClick>>", lambda e: self._activate(self._fm))

        self._fn = ExprField(f, label="  N(x, y)  →  coeficiente de dy")
        self._fn.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._fn.bind("<<FieldClick>>", lambda e: self._activate(self._fn))

        ctk.CTkLabel(f, text="  Ejemplos predefinidos:",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=4, column=0, sticky="w", pady=(0, 4))

        examples = [
            ("M = x²+y²    N = −2xy   ← Caso de aplicación CPU",
             [("x**2","x²","var"),("+","+","op"),("y**2","y²","var")],
             [("-","−","_neg"),("2","2","d"),("x","x","var"),("y","y","var")]),
            ("M = 2xy       N = y²−x²",
             [("2","2","d"),("x","x","var"),("y","y","var")],
             [("y**2","y²","var"),("-","-","op"),("x**2","x²","var")]),
            ("M = x³+y³    N = x²y",
             [("x**3","x³","var"),("+","+","op"),("y**3","y³","var")],
             [("x**2*y","x²y","var")]),
            ("M = x²+xy    N = y²",
             [("x**2","x²","var"),("+","+","op"),("x","x","var"),("y","y","var")],
             [("y**2","y²","var")]),
        ]
        for i, (lbl, mt, nt) in enumerate(examples):
            ctk.CTkButton(f, text=lbl, height=32, corner_radius=6,
                           fg_color=CARD, hover_color=BDR,
                           font=ctk.CTkFont("Consolas", 11), text_color=DIM,
                           border_width=1, border_color=BDR,
                           command=lambda m=mt, n=nt: self._load_t1(m, n)
                           ).grid(row=5 + i, column=0, pady=2, sticky="ew")

        self._activate(self._fm)
        self._prev.set_t1("", "")

    def _load_t1(self, mt, nt):
        self._fm.set_tokens(mt)
        self._fn.set_tokens(nt)
        self._preview()

    # ─────────────────────────────────────────────────────────────────────────
    #  INPUTS — TIPO 2  (teclado físico habilitado)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_t2_inputs(self):
        self._clear_inp_area()
        f = self._inp

        ctk.CTkLabel(f, text="  a · y″  +  b · y′  +  c · y  =  0",
                     font=ctk.CTkFont("Consolas", 12, "bold"),
                     text_color=CYAN).grid(row=0, column=0, sticky="w", pady=(4, 2))

        # Instrucciones mejoradas
        instr = ctk.CTkFrame(f, fg_color="#0d2a1e", corner_radius=8,
                              border_width=1, border_color="#1a4a2e")
        instr.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        instr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(instr,
                     text="  ① Haz clic en la tarjeta del coeficiente (a, b o c)\n"
                          "  ② Escribe el valor con el teclado físico o el de pantalla\n"
                          "  ③ Solo se aceptan números enteros o decimales (ej: 1, -2, 0.5)\n"
                          "  ④ El campo activo se resalta en azul",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=GREEN,
                     justify="left"
                     ).grid(row=0, column=0, padx=8, pady=6, sticky="w")

        cf = ctk.CTkFrame(f, fg_color="transparent")
        cf.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        cf.grid_columnconfigure((0, 1, 2), weight=1)

        self._t2f = {}
        coefs = [("a", "y″", ACC,    "(2do orden)"),
                 ("b", "y′", PURPLE, "(1er orden)"),
                 ("c", "y",  GREEN,  "(término ind.)")]

        for col, (key, deriv, color, sub) in enumerate(coefs):
            fr = ctk.CTkFrame(cf, fg_color=CARD, corner_radius=10,
                               border_width=2, border_color=BDR)
            fr.grid(row=0, column=col, padx=4, sticky="nsew")
            fr.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(fr, text=key,
                         font=ctk.CTkFont("Consolas", 24, "bold"),
                         text_color=color).grid(row=0, column=0, pady=(10, 0))
            ctk.CTkLabel(fr, text=f"de {deriv}",
                         font=ctk.CTkFont("Segoe UI", 9), text_color=DIM
                         ).grid(row=1, column=0, pady=(0, 2))
            ctk.CTkLabel(fr, text=sub,
                         font=ctk.CTkFont("Segoe UI", 8), text_color=DIM
                         ).grid(row=2, column=0, pady=(0, 4))

            ef = ExprField(fr, placeholder="0")
            ef.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 10))
            ef.bind("<<FieldClick>>", lambda e, ef=ef, fr=fr: self._activate_t2(ef, fr))

            # CORRECCIÓN: bind en la tarjeta también
            fr.bind("<Button-1>", lambda e, ef=ef, fr=fr: self._activate_t2(ef, fr))

            self._t2f[key] = ef
            self._t2f[f"_{key}_frame"] = fr   # guardar frame para resaltar

        # Teclado físico: bind global a la ventana
        self.bind("<Key>", self._on_key_press)

        ctk.CTkLabel(f, text="  Ejemplos predefinidos:",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=3, column=0, sticky="w", pady=(0, 4))

        examples = [
            ("y″ − 5y′ + 6y = 0     (raíces reales distintas)",  1, -5,  6),
            ("y″ − 6y′ + 9y = 0     (raíz repetida)",            1, -6,  9),
            ("y″ + 4y = 0            (complejas puras)",           1,  0,  4),
            ("y″ + 2y′ + 5y = 0     (sub-amortiguado)",           1,  2,  5),
            ("y″ + 2y′ + 4y = 0     ← Caso RLC de aplicación",   1,  2,  4),
        ]
        for i, (lbl, a, b, c) in enumerate(examples):
            ctk.CTkButton(f, text=lbl, height=32, corner_radius=6,
                           fg_color=CARD, hover_color=BDR,
                           font=ctk.CTkFont("Consolas", 11), text_color=DIM,
                           border_width=1, border_color=BDR,
                           command=lambda a=a, b=b, c=c: self._load_t2(a, b, c)
                           ).grid(row=4 + i, column=0, pady=2, sticky="ew")

        self._activate_t2(self._t2f["a"], self._t2f["_a_frame"])
        self._prev.set_t2("", "", "")

    def _activate_t2(self, ef: "ExprField", fr: ctk.CTkFrame):
        """Activa un campo de tipo 2 y resalta su tarjeta."""
        # Desactivar el anterior
        if self._active_field and self._active_field is not ef:
            self._active_field.activate(False)
            # Desresaltar tarjetas previas
            for k in ("a", "b", "c"):
                if f"_{k}_frame" in self._t2f:
                    self._t2f[f"_{k}_frame"].configure(border_color=BDR)
        self._active_field = ef
        ef.activate(True)
        fr.configure(border_color=BACT)

    def _on_key_press(self, event):
        """Teclado físico para tipo 2: solo dígitos, punto y signo menos."""
        if self._eq_type != "tipo2" or not self._active_field:
            return
        ch = event.char
        keysym = event.keysym
        if keysym == "BackSpace":
            self._active_field.backspace(); self._preview(); return
        if keysym in ("minus", "KP_Subtract") or ch == "-":
            self._active_field.negate(); self._preview(); return
        if ch.isdigit():
            self._active_field.push(ch, ch, "d"); self._preview(); return
        if ch == ".":
            self._active_field.push(".", ".", "d"); self._preview(); return
        if keysym == "Tab":
            # Tab navega entre a → b → c → a
            order = ["a", "b", "c"]
            cur = None
            for k in order:
                if self._t2f[k] is self._active_field:
                    cur = k; break
            if cur:
                nxt = order[(order.index(cur) + 1) % 3]
                self._activate_t2(self._t2f[nxt], self._t2f[f"_{nxt}_frame"])
            return

    def _load_t2(self, a, b, c):
        for key, val in [("a", a), ("b", b), ("c", c)]:
            toks = []
            if val < 0:
                sv = str(abs(int(val))) if val == int(val) else str(abs(val))
                toks = [("-", "−", "_neg"), (sv, sv, "d")]
            elif val != 0:
                sv = str(int(val)) if val == int(val) else str(val)
                toks = [(sv, sv, "d")]
            self._t2f[key].set_tokens(toks)
        self._preview()

    def _clear_inp_area(self):
        for w in self._inp.winfo_children(): w.destroy()
        self._active_field = None
        # Desconectar teclado físico al limpiar
        try:
            self.unbind("<Key>")
        except Exception:
            pass

    def _activate(self, fld: "ExprField"):
        """Activar campo (Tipo 1)."""
        if self._active_field and self._active_field is not fld:
            self._active_field.activate(False)
        self._active_field = fld
        fld.activate(True)

    def _preview(self):
        if self._eq_type == "tipo1":
            m = self._fm.get_disp() if hasattr(self, "_fm") else ""
            n = self._fn.get_disp() if hasattr(self, "_fn") else ""
            self._prev.set_t1(m, n)
        else:
            vals = {k: self._t2f[k].get_disp() for k in "abc"} if hasattr(self, "_t2f") else {}
            self._prev.set_t2(vals.get("a",""), vals.get("b",""), vals.get("c",""))

    # ─────────────────────────────────────────────────────────────────────────
    #  TECLADO MATEMÁTICO
    # ─────────────────────────────────────────────────────────────────────────
    def _build_keyboard(self):
        for w in self._kbd.winfo_children(): w.destroy()
        k = self._kbd
        k.grid_columnconfigure(0, weight=1)
        row = 0

        def mkbtn(parent, txt, cmd, r, c, span=1, h=42,
                  fg=CARD, hv=BDR, tc=TEXT, bold=False):
            ctk.CTkButton(
                parent, text=txt, height=h,
                font=ctk.CTkFont("Consolas", 13, "bold" if bold else "normal"),
                corner_radius=7, fg_color=fg, hover_color=hv, text_color=tc,
                command=cmd
            ).grid(row=r, column=c, columnspan=span, padx=3, pady=3, sticky="ew")

        if self._eq_type == "tipo1":
            lf = ctk.CTkFrame(k, fg_color="transparent")
            lf.grid(row=row, column=0, padx=10, pady=(12, 4), sticky="ew"); row += 1
            ctk.CTkLabel(lf, text="  Términos en x e y",
                         font=ctk.CTkFont("Segoe UI", 10, "bold"),
                         text_color=DIM).pack(anchor="w")

            vf = ctk.CTkFrame(k, fg_color="#0d2035", corner_radius=8)
            vf.grid(row=row, column=0, padx=10, pady=(0, 6), sticky="ew"); row += 1
            for i in range(6): vf.grid_columnconfigure(i, weight=1)

            for idx, (lbl, intl, disp, kind) in enumerate(VARS_T1):
                r2, c2 = divmod(idx, 6)
                mkbtn(vf, lbl,
                      lambda i=intl, d=disp, kk=kind: self._kpush(i, d, kk),
                      r2, c2, fg="#0d2035", hv="#1a3d5c", tc=CYAN, bold=True, h=40)

            ctk.CTkFrame(k, height=1, fg_color=BDR).grid(
                row=row, column=0, padx=10, pady=4, sticky="ew"); row += 1

        # Sección numérica
        if self._eq_type == "tipo1":
            hint = "  Números y operadores"
        else:
            hint = "  Ingresa el coeficiente  (también puedes usar el teclado físico — Tab para pasar al siguiente)"

        lf2 = ctk.CTkFrame(k, fg_color="transparent")
        lf2.grid(row=row, column=0, padx=10, pady=(6, 4), sticky="ew"); row += 1
        ctk.CTkLabel(lf2, text=hint,
                     font=ctk.CTkFont("Segoe UI", 10, "bold"),
                     text_color=DIM).pack(anchor="w")

        nf = ctk.CTkFrame(k, fg_color="transparent")
        nf.grid(row=row, column=0, padx=10, pady=(0, 12), sticky="ew"); row += 1
        for i in range(5): nf.grid_columnconfigure(i, weight=1)

        NUM_ROWS = [
            [("7","7","d"), ("8","8","d"), ("9","9","d"), ("+","+","op"), ("−","-","op")],
            [("4","4","d"), ("5","5","d"), ("6","6","d"), (".",".", "d"), ("⌫","", "back")],
            [("1","1","d"), ("2","2","d"), ("3","3","d"), ("±","", "neg"), ("✕","","clr")],
            [("0","0","d"),],
        ]
        for ri, rowdata in enumerate(NUM_ROWS):
            for ci, (lbl, val, kind) in enumerate(rowdata):
                if kind == "back":
                    mkbtn(nf, "⌫  Borrar", self._kback, ri, ci,
                          fg="#2a1020", hv="#4a2030", tc="#ff9999", h=42)
                elif kind == "neg":
                    mkbtn(nf, "± cambiar signo", self._kneg, ri, ci,
                          fg=CARD, hv=BDR, tc=DIM, h=42)
                elif kind == "clr":
                    mkbtn(nf, "✕  Borrar campo", self._kclr, ri, ci,
                          fg="#2a1020", hv="#4a2030", tc="#ff9999", h=42)
                elif kind == "op":
                    mkbtn(nf, lbl,
                          lambda v=val, l=lbl: self._kpush(v, l, "op"),
                          ri, ci, fg="#0d2a1e", hv="#1a4a2e", tc=GREEN, bold=True, h=42)
                else:
                    span = 5 if lbl == "0" else 1
                    mkbtn(nf, lbl,
                          lambda v=val, l=lbl: self._kpush(v, l, "d"),
                          ri, ci, span=span, h=42)

    def _kpush(self, i, d, k):
        if self._active_field:
            self._active_field.push(i, d, k); self._preview()

    def _kback(self):
        if self._active_field:
            self._active_field.backspace(); self._preview()

    def _kneg(self):
        if self._active_field:
            self._active_field.negate(); self._preview()

    def _kclr(self):
        if self._active_field:
            self._active_field.clear(); self._preview()

    def _on_type_change(self, val: str):
        self._eq_type = "tipo1" if "1" in val else "tipo2"
        if self._eq_type == "tipo1":
            self._type_sub.configure(text="  M(x,y)·dx + N(x,y)·dy = 0")
            self._build_t1_inputs()
        else:
            self._type_sub.configure(text="  ay″ + by′ + cy = 0")
            self._build_t2_inputs()
        self._build_keyboard()
        self._update_app_tab()

    # ─────────────────────────────────────────────────────────────────────────
    #  PANEL DERECHO
    # ─────────────────────────────────────────────────────────────────────────
    def _build_right(self, p):
        ctk.CTkLabel(p,
                     text="Solucionador de Ecuaciones Diferenciales Homogéneas",
                     font=ctk.CTkFont("Segoe UI", 19, "bold"),
                     text_color=TEXT
                     ).grid(row=0, column=0, sticky="w", pady=(0, 14))

        self._tabs = ctk.CTkTabview(
            p, anchor="nw",
            segmented_button_fg_color=CARD,
            segmented_button_selected_color=ACC,
            segmented_button_selected_hover_color="#2879f0",
            fg_color=CARD, corner_radius=12)
        self._tabs.grid(row=1, column=0, sticky="nsew")

        self._tab_app  = self._tabs.add("  Problema de Aplicación  ")
        self._tab_proc = self._tabs.add("  Solución Paso a Paso  ")
        self._tab_graf = self._tabs.add("  Gráfica de Solución  ")

        for t in (self._tab_app, self._tab_proc, self._tab_graf):
            t.grid_columnconfigure(0, weight=1)
            t.grid_rowconfigure(0, weight=1)

        # Pestaña solución
        self._out = ctk.CTkTextbox(
            self._tab_proc,
            font=ctk.CTkFont("Consolas", 13),
            wrap="word", corner_radius=8,
            fg_color=INP, text_color=TEXT,
            scrollbar_button_color=BDR,
            scrollbar_button_hover_color=ACC)
        self._out.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Pestaña gráfica
        self._fig = Figure(figsize=(7, 5), dpi=100)
        self._fig.patch.set_facecolor(INP)
        self._cv  = FigureCanvasTkAgg(self._fig, master=self._tab_graf)
        self._cv.get_tk_widget().configure(bg=INP)
        self._cv.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Pestaña aplicación
        self._app_scroll = ctk.CTkScrollableFrame(
            self._tab_app, fg_color="transparent",
            scrollbar_button_color=BDR,
            scrollbar_button_hover_color=ACC)
        self._app_scroll.grid(row=0, column=0, sticky="nsew")
        self._app_scroll.grid_columnconfigure(0, weight=1)

        self._show_welcome()
        self._blank_graph()
        self._update_app_tab()

    # ─────────────────────────────────────────────────────────────────────────
    #  PESTAÑA DE APLICACIÓN
    # ─────────────────────────────────────────────────────────────────────────
    def _update_app_tab(self):
        for w in self._app_scroll.winfo_children(): w.destroy()
        if self._eq_type == "tipo1":
            self._build_app_t1()
        else:
            self._build_app_t2()

    def _acard(self, parent, row):
        f = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10,
                          border_width=1, border_color=BDR)
        f.grid(row=row, column=0, padx=8, pady=6, sticky="ew")
        f.grid_columnconfigure(0, weight=1)
        return f

    def _atitle(self, parent, text, row, color=ACC):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color=color).grid(row=row, column=0, padx=14, pady=(12, 4), sticky="w")

    def _abody(self, parent, text, row):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont("Segoe UI", 11), text_color=DIM,
                     justify="left", wraplength=720
                     ).grid(row=row, column=0, padx=14, pady=(0, 8), sticky="w")

    def _build_app_t1(self):
        af = self._app_scroll

        ban = ctk.CTkFrame(af, fg_color="#0d1f33", corner_radius=12,
                            border_width=1, border_color=CYAN)
        ban.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="ew")
        ban.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ban,
                     text="Líneas de flujo en sistemas de enfriamiento líquido para CPUs",
                     font=ctk.CTkFont("Segoe UI", 15, "bold"),
                     text_color=CYAN).grid(row=0, column=0, padx=16, pady=(14, 2), sticky="w")
        ctk.CTkLabel(ban,
                     text="Aplicación de ED homogéneas de 1° orden en ingeniería de sistemas computacionales",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

        fig_a = Figure(figsize=(7, 3.1), dpi=96)
        fig_a.patch.set_facecolor(CARD)
        ax = fig_a.add_subplot(111)
        self._draw_cooler(ax)
        cva = FigureCanvasTkAgg(fig_a, master=af)
        cva.get_tk_widget().configure(bg=CARD)
        cva.get_tk_widget().grid(row=1, column=0, padx=8, pady=4, sticky="ew")
        cva.draw()

        c1 = self._acard(af, 2)
        self._atitle(c1, "Contexto del problema", 0, CYAN)
        self._abody(c1,
            "En los procesadores modernos (CPUs y GPUs) de alta densidad, el calor generado puede "
            "superar los 100 W/cm². Los sistemas de enfriamiento líquido utilizan un fluido refrigerante "
            "que circula por microcanales grabados sobre el disipador del procesador.", 1)
        self._abody(c1,
            "Las trayectorias de las partículas de fluido se denominan líneas de corriente (streamlines). "
            "Conocer estas curvas es fundamental para el diseño de microcanales y la optimización del "
            "intercambio térmico.", 2)

        c2 = self._acard(af, 3)
        self._atitle(c2, "Modelo matemático — ¿Por qué aparece una ED homogénea?", 0)
        self._abody(c2,
            "La condición de tangencia entre la trayectoria y el vector velocidad da:", 1)
        ctk.CTkLabel(c2, text="  dy/dx  =  − M(x,y) / N(x,y)   ⟺   M(x,y)·dx + N(x,y)·dy = 0",
                     font=ctk.CTkFont("Consolas", 13, "bold"),
                     text_color=GREEN).grid(row=2, column=0, padx=14, pady=6, sticky="w")
        self._abody(c2,
            "Para un patrón de flujo real en un disipador de CPU, el análisis de CFD determinó:", 3)
        ctk.CTkLabel(c2, text="  M(x, y) = x² + y²        N(x, y) = −2xy",
                     font=ctk.CTkFont("Consolas", 13),
                     text_color=YELLOW).grid(row=4, column=0, padx=14, pady=4, sticky="w")
        self._abody(c2,
            "Ambas funciones son homogéneas de grado 2. La solución resulta ser la familia de "
            "parábolas x² = C·y, que describe las trayectorias del refrigerante.", 5)

        c3 = self._acard(af, 4)
        self._atitle(c3, "Ecuación que resuelve el solucionador", 0, GREEN)
        ctk.CTkLabel(c3, text="  M(x,y) = x² + y²       N(x,y) = −2xy",
                     font=ctk.CTkFont("Consolas", 14, "bold"),
                     text_color=CYAN).grid(row=1, column=0, padx=14, pady=6, sticky="w")
        ctk.CTkLabel(c3, text="  Solución esperada:  x² = C·y  (familia de parábolas)",
                     font=ctk.CTkFont("Consolas", 11),
                     text_color=DIM).grid(row=2, column=0, padx=14, pady=(0, 4), sticky="w")
        ctk.CTkButton(c3, text="  Cargar este ejemplo en el solucionador",
                       height=40, corner_radius=8,
                       fg_color=ACC, hover_color="#2879f0",
                       font=ctk.CTkFont("Segoe UI", 12),
                       command=self._load_app1
                       ).grid(row=3, column=0, padx=14, pady=(4, 14), sticky="w")

    def _load_app1(self):
        self._load_t1(
            [("x**2","x²","var"),("+","+","op"),("y**2","y²","var")],
            [("-","−","_neg"),("2","2","d"),("x","x","var"),("y","y","var")])
        self._tabs.set("  Solución Paso a Paso  ")

    def _draw_cooler(self, ax):
        ax.set_xlim(0, 12); ax.set_ylim(0, 5.5)
        ax.set_facecolor("#0d1f33"); ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0.5, 0.3), 11, 4.8,
            boxstyle="round,pad=0.15", fc="#101a2e", ec="#30363d", lw=2))
        for xi in np.linspace(0.9, 10.6, 10):
            ax.add_patch(mpatches.FancyBboxPatch((xi, 0.5), 0.55, 4.2,
                boxstyle="square", fc="#0a2a4a", ec="#1a3a6a", lw=1, alpha=0.8))
        for C_v in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
            xv = np.linspace(-4.5, 4.5, 600)
            yv = xv**2 / C_v
            mask = (yv >= 0.4) & (yv <= 4.9)
            xp = xv[mask] * 1.2 + 6.0
            yp = yv[mask] * 0.93 + 0.35
            ok = (xp >= 0.6) & (xp <= 11.4)
            if ok.sum() > 3:
                ax.plot(xp[ok], yp[ok], color="#79c0ff", alpha=0.6, lw=1.6, zorder=4)
        for xf, yf, xd, yd in [(7.3,3.2,8.0,2.7),(5.0,2.1,4.4,2.5),(6.2,1.2,6.7,1.6)]:
            ax.annotate("", xy=(xd,yd), xytext=(xf,yf),
                         arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.4))
        ax.text(6, 5.3, "Streamlines del refrigerante:  x² = C·y  (parábolas)",
                ha="center", color=CYAN, fontsize=9, fontweight="bold")
        ax.text(6, 0.0, "ED:  ( x² + y² ) dx  −  2xy · dy  =  0",
                ha="center", color=YELLOW, fontsize=9, fontweight="bold")
        ax.set_title("Disipador de calor — Microcanales y trayectorias del refrigerante",
                     color=TEXT, fontsize=10, pad=6)

    def _build_app_t2(self):
        af = self._app_scroll

        ban = ctk.CTkFrame(af, fg_color="#150e2e", corner_radius=12,
                            border_width=1, border_color=PURPLE)
        ban.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="ew")
        ban.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ban,
                     text="Circuito RLC en serie — Respuesta transitoria de una señal",
                     font=ctk.CTkFont("Segoe UI", 15, "bold"),
                     text_color=PURPLE).grid(row=0, column=0, padx=16, pady=(14, 2), sticky="w")
        ctk.CTkLabel(ban,
                     text="Aplicación de ED lineal homogénea de 2° orden en análisis de circuitos electrónicos",
                     font=ctk.CTkFont("Segoe UI", 10), text_color=DIM
                     ).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

        fig_b = Figure(figsize=(7, 3.1), dpi=96)
        fig_b.patch.set_facecolor(CARD)
        ax = fig_b.add_subplot(111)
        self._draw_rlc(ax)
        cvb = FigureCanvasTkAgg(fig_b, master=af)
        cvb.get_tk_widget().configure(bg=CARD)
        cvb.get_tk_widget().grid(row=1, column=0, padx=8, pady=4, sticky="ew")
        cvb.draw()

        c1 = self._acard(af, 2)
        self._atitle(c1, "Contexto del problema", 0, PURPLE)
        self._abody(c1,
            "Los circuitos RLC en serie son componentes fundamentales en sistemas de comunicación "
            "digital: tarjetas de red, módems, interfaces HDMI y USB implementan filtros RLC para "
            "eliminar ruido y aislar frecuencias de operación.", 1)

        c2 = self._acard(af, 3)
        self._atitle(c2, "Modelo matemático — Ley de Voltajes de Kirchhoff", 0)
        ctk.CTkLabel(c2, text="  L · q″(t)  +  R · q′(t)  +  (1/C) · q(t)  =  0",
                     font=ctk.CTkFont("Consolas", 14, "bold"),
                     text_color=GREEN).grid(row=1, column=0, padx=14, pady=6, sticky="w")
        self._abody(c2,
            "Esta es exactamente la forma  ay″ + by′ + cy = 0  con  a = L,  b = R,  c = 1/C.\n\n"
            "El discriminante Δ = R² − 4L/C determina el comportamiento eléctrico:\n"
            "  • Δ > 0  →  Sobre-amortiguado: decae sin oscilar (R grande)\n"
            "  • Δ = 0  →  Críticamente amortiguado: retorno más rápido sin oscilación\n"
            "  • Δ < 0  →  Sub-amortiguado: oscilaciones amortiguadas (filtro resonante)", 2)

        c3 = self._acard(af, 4)
        self._atitle(c3, "Ejemplo concreto — Filtro de señal digital", 0, GREEN)
        ctk.CTkLabel(c3, text="  1·q″  +  2·q′  +  4·q  =  0",
                     font=ctk.CTkFont("Consolas", 14, "bold"),
                     text_color=CYAN).grid(row=1, column=0, padx=14, pady=4, sticky="w")
        self._abody(c3,
            "Δ = 4 − 16 = −12 < 0  →  sub-amortiguado.\n"
            "q(t) = e^(−t)[C₁cos(√3·t) + C₂sin(√3·t)]", 2)
        ctk.CTkButton(c3, text="  Cargar este ejemplo en el solucionador",
                       height=40, corner_radius=8,
                       fg_color=ACC, hover_color="#2879f0",
                       font=ctk.CTkFont("Segoe UI", 12),
                       command=lambda: self._load_t2(1, 2, 4)
                       ).grid(row=3, column=0, padx=14, pady=(4, 14), sticky="w")

    def _draw_rlc(self, ax):
        ax.set_xlim(0, 10); ax.set_ylim(-0.5, 4.5)
        ax.set_facecolor(CARD); ax.axis("off")
        y_top, y_bot = 3.8, 0.8
        x0, xR1, xR2, xL1, xL2, xC1, xC2, xn = 0.5, 1.5, 3.5, 4.2, 6.2, 6.9, 8.9, 9.5
        ax.plot([x0, xR1], [y_top, y_top], color=TEXT, lw=2.2)
        ax.plot([xR2, xL1], [y_top, y_top], color=TEXT, lw=2.2)
        ax.plot([xL2, xC1], [y_top, y_top], color=TEXT, lw=2.2)
        ax.plot([xC2, xn],  [y_top, y_top], color=TEXT, lw=2.2)
        ax.plot([xn,  xn],  [y_top, y_bot], color=TEXT, lw=2.2)
        ax.plot([x0,  x0],  [y_top, y_bot], color=TEXT, lw=2.2)
        ax.plot([x0,  xn],  [y_bot, y_bot], color=TEXT, lw=2.2)
        nz = 8
        rxv = np.linspace(xR1, xR2, nz*2+2)
        ryv = np.array([y_top+(0.22 if i%2==1 else -0.22) for i in range(len(rxv))])
        ryv[0] = ryv[-1] = y_top
        ax.plot(rxv, ryv, color=YELLOW, lw=2.0)
        ax.text((xR1+xR2)/2, y_top+0.52, "R", ha="center", color=YELLOW, fontsize=15, fontweight="bold")
        ax.text((xR1+xR2)/2, y_top-0.42, "Resistencia", ha="center", color=DIM, fontsize=8)
        nc = 5
        lxv, lyv = [], []
        for i in range(nc):
            theta = np.linspace(0, np.pi, 30)
            r_i = (xL2-xL1)/(2*nc)
            cx = xL1+(2*i+1)*r_i
            lxv.extend(list(cx+r_i*np.cos(theta[::-1])))
            lyv.extend(list(y_top+r_i*np.sin(theta)))
        ax.plot(lxv, lyv, color=CYAN, lw=2.0)
        ax.text((xL1+xL2)/2, y_top+0.52, "L", ha="center", color=CYAN, fontsize=15, fontweight="bold")
        ax.text((xL1+xL2)/2, y_top-0.42, "Inductor", ha="center", color=DIM, fontsize=8)
        cx = (xC1+xC2)/2; cgap = 0.22
        ax.plot([xC1, cx-0.01], [y_top, y_top], color=TEXT, lw=2.2)
        ax.plot([cx-0.01, cx-0.01], [y_top-0.45, y_top+0.45], color=GREEN, lw=5)
        ax.plot([cx+cgap, cx+cgap], [y_top-0.45, y_top+0.45], color=GREEN, lw=5)
        ax.plot([cx+cgap, xC2], [y_top, y_top], color=TEXT, lw=2.2)
        ax.text((xC1+xC2)/2, y_top+0.52, "C", ha="center", color=GREEN, fontsize=15, fontweight="bold")
        ax.text((xC1+xC2)/2, y_top-0.42, "Capacitor", ha="center", color=DIM, fontsize=8)
        ax.annotate("", xy=(5.5, y_top), xytext=(4.5, y_top),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
        ax.text(5.0, y_top+0.65, "i(t) = q′(t)", ha="center", color=RED, fontsize=9, fontstyle="italic")
        ax.text(5.0, y_bot-0.35,
                "L·q″(t)  +  R·q′(t)  +  (1/C)·q(t)  =  0",
                ha="center", color=TEXT, fontsize=10, fontweight="bold")
        ax.set_title("Circuito RLC en serie — Modelo de filtro de señal digital",
                     color=TEXT, fontsize=10, pad=6)

    # ─────────────────────────────────────────────────────────────────────────
    #  SOLUCIONADOR
    # ─────────────────────────────────────────────────────────────────────────
    def _solve(self):
        self._btn_solve.configure(state="disabled", text="  Calculando...")
        self.update()
        try:
            if self._eq_type == "tipo1":
                self._solve_t1()
            else:
                self._solve_t2()
        except Exception as ex:
            self._write_error(f"Error inesperado: {ex}")
        finally:
            self._btn_solve.configure(state="normal", text="▶  Resolver paso a paso")

    # ─────────────────────────────────────────────────────────────────────────
    #  SOLVER TIPO 1
    # ─────────────────────────────────────────────────────────────────────────
    def _solve_t1(self):
        me = self._fm.get_expr().strip()
        ne = self._fn.get_expr().strip()
        md = self._fm.get_disp().strip()
        nd = self._fn.get_disp().strip()

        if not me or not ne:
            self._write_error("Ingresa M(x,y) y N(x,y) usando el teclado."); return
        try:
            M = sp.sympify(me, locals={"x": xs, "y": ys})
            N = sp.sympify(ne, locals={"x": xs, "y": ys})
        except Exception as e:
            self._write_error(
                f"Expresión no válida: {e}\n\n"
                f"  Expresión M recibida: {repr(me)}\n"
                f"  Expresión N recibida: {repr(ne)}\n\n"
                "  Usa el teclado de la pantalla para ingresar los términos."
            ); return

        HL = "─" * 62
        L  = []

        L += [
            "SOLUCIÓN — Ecuación Diferencial Homogénea de Primer Orden",
            "=" * 62, "",
            "  Ecuación dada:",
            f"    ( {md} ) dx  +  ( {nd} ) dy  =  0",
            "",
        ]

        # Paso 1: Homogeneidad
        L += [HL, "  Paso 1 · Verificación de homogeneidad", HL, ""]
        Mt = sp.expand(M.subs([(xs, ts*xs), (ys, ts*ys)]))
        Nt = sp.expand(N.subs([(xs, ts*xs), (ys, ts*ys)]))
        g  = None
        for n in range(8):
            try:
                Ms = sp.simplify(Mt / ts**n)
                Ns = sp.simplify(Nt / ts**n)
                if not Ms.has(ts) and not Ns.has(ts):
                    g = n; break
            except Exception:
                continue

        if g is None:
            L += [
                "  La ecuación NO es homogénea.",
                "",
                "  Para ser homogénea, M y N deben satisfacer:",
                "    M(tx, ty) = tⁿ · M(x,y)   y   N(tx, ty) = tⁿ · N(x,y)",
                "",
                "  Verifica que todos los términos de M y N tengan el mismo grado total.",
            ]
            self._write_out(L); return

        L += [
            "  Una función f(x,y) es homogénea de grado n si:",
            "    f(tx, ty) = tⁿ · f(x,y)    para todo t > 0",
            "",
            f"  M(tx, ty) = t^{g} · M(x, y)  ✓",
            f"  N(tx, ty) = t^{g} · N(x, y)  ✓",
            "",
            f"  Conclusión: ambas funciones son homogéneas de grado n = {g}.",
            f"  La ecuación es HOMOGÉNEA de grado {g}.",
            "  Se aplicará la sustitución y = vx.", "",
        ]

        # Paso 2: Sustitución
        L += [HL, "  Paso 2 · Sustitución homogénea  y = vx", HL, ""]
        L += [
            "    y  =  v · x       donde v = v(x)",
            "    dy =  v · dx  +  x · dv",
            "",
            "  Esta sustitución convierte la ED en separable en v y x.", "",
        ]

        # Paso 3: Sustitución en la ED
        L += [HL, "  Paso 3 · Sustitución en la ecuación diferencial", HL, ""]
        Mv  = sp.simplify(M.subs(ys, vs * xs))
        Nv  = sp.simplify(N.subs(ys, vs * xs))
        tot = sp.simplify(Mv + vs * Nv)

        L += [
            "  Reemplazando y = vx en M y N:",
            f"    M(x, vx) = {sp.factor(Mv)}",
            f"    N(x, vx) = {sp.factor(Nv)}",
            "",
            "  Sustituyendo dy = v·dx + x·dv:",
            f"    [ {sp.factor(tot)} ] dx  +  [ {sp.factor(Nv)} ]·x·dv = 0",
            "",
        ]

        # Paso 4: Separación
        L += [HL, "  Paso 4 · Separación de variables", HL, ""]
        M1 = sp.simplify(M.subs([(xs, 1), (ys, vs)]))
        N1 = sp.simplify(N.subs([(xs, 1), (ys, vs)]))
        A  = sp.simplify(M1 + vs * N1)
        B  = sp.simplify(N1)

        if A == 0:
            L += ["  El coeficiente A(v) resultó ser 0. Verifica la ecuación."]
            self._write_out(L); return

        ig = sp.simplify(-B / A)
        L += [
            f"  Factorizando x^{g} y dividiendo entre x^{g+1}:",
            "",
            "  Forma separable:",
            f"    dx/x   =   ( {sp.factor(ig)} ) dv", "",
        ]

        # Paso 5: Integración
        L += [HL, "  Paso 5 · Integración de ambos miembros", HL, ""]
        try:
            itg = sp.simplify(sp.integrate(ig, vs))
            L += [
                f"    ∫ dx/x   =   ∫ ( {sp.factor(ig)} ) dv",
                "",
                f"    ln|x|   =   {itg}  +  C₁", "",
            ]
        except Exception:
            L += ["  (La integral no pudo resolverse simbólicamente.)"]
            self._write_out(L); return

        # Paso 6: Retrosustitución
        L += [HL, "  Paso 6 · Retrosustitución  v = y/x", HL, ""]
        try:
            sol = sp.simplify(itg.subs(vs, ys / xs))
            L += [
                "  Reemplazando v = y/x:",
                "",
                "  ┌─────────────────────────────────────────────────────────┐",
                f"  │   ln|x|  =  {str(sol):<41s}  +  C   │",
                "  └─────────────────────────────────────────────────────────┘",
                "",
                "  C es la constante arbitraria de integración.",
                "  Con condición inicial y(x₀) = y₀ se determina C.", "",
                "=" * 62,
            ]
        except Exception:
            L += [f"  Sustituye v = y/x en:   ln|x| = {itg} + C", "", "=" * 62]

        self._write_out(L)
        # CORRECCIÓN: primero mostrar solución, gráfica en segundo plano
        self._plot_t1(M, N)

    # ─────────────────────────────────────────────────────────────────────────
    #  SOLVER TIPO 2
    # ─────────────────────────────────────────────────────────────────────────
    def _solve_t2(self):
        try:
            af = float(self._t2f["a"].get_expr() or "0")
            bf = float(self._t2f["b"].get_expr() or "0")
            cf = float(self._t2f["c"].get_expr() or "0")
        except ValueError:
            self._write_error("Ingresa solo valores numéricos en a, b y c.\n"
                              "Ejemplos válidos: 1, -2, 0.5"); return

        if af == 0:
            self._write_error("El coeficiente 'a' no puede ser cero (no sería de 2° orden)."); return

        a = sp.Rational(af).limit_denominator(1000)
        b = sp.Rational(bf).limit_denominator(1000)
        c = sp.Rational(cf).limit_denominator(1000)

        HL = "─" * 62
        L  = []

        L += [
            "SOLUCIÓN — ED Homogénea Lineal de Segundo Orden",
            "=" * 62, "",
            "  Ecuación dada:",
            f"    {self._fmt_eq(af, bf, cf)}  =  0",
            "",
        ]

        # Paso 1: Coeficientes
        L += [HL, "  Paso 1 · Identificación de coeficientes", HL, ""]
        L += [
            f"    a = {a}    →  coeficiente de y″ (2a derivada)",
            f"    b = {b}    →  coeficiente de y′ (1a derivada)",
            f"    c = {c}    →  coeficiente de y  (término sin derivar)", "",
        ]

        # Paso 2: Ecuación característica
        L += [HL, "  Paso 2 · Ecuación característica", HL, ""]
        L += [
            "  Proponemos  y = e^(rt):",
            "    y′ = r·e^(rt)         y″ = r²·e^(rt)",
            "",
            "  Sustituyendo y factorizando e^(rt) ≠ 0:",
            "",
            f"    {self._fmt_char(af, bf, cf)}  =  0",
            "",
        ]

        # Paso 3: Discriminante
        disc = b**2 - 4*a*c
        df   = float(disc)

        L += [HL, "  Paso 3 · Cálculo del discriminante", HL, ""]
        L += [
            "  Δ = b² − 4ac",
            f"  Δ = ({b})² − 4·({a})·({c})",
            f"  Δ = {b**2} − {4*a*c}",
            f"  Δ = {disc}", "",
        ]

        # Paso 4: Raíces
        r1s = sp.simplify((-b + sp.sqrt(disc)) / (2*a))
        r2s = sp.simplify((-b - sp.sqrt(disc)) / (2*a))

        L += [HL, "  Paso 4 · Cálculo de las raíces", HL, ""]
        L += [
            "  r = ( −b ± √Δ ) / (2a)",
            "",
            f"  r₁ = {r1s}",
            f"  r₂ = {r2s}", "",
        ]

        # Paso 5: Caso y solución
        L += [HL, "  Paso 5 · Caso y solución general", HL, ""]

        if df > 0:
            cs = "dist"
            r1n, r2n = float(r1s), float(r2s)
            alp = bet = None
            L += [
                f"  Δ = {disc} > 0   →   CASO 1: Raíces reales y distintas",
                "",
                f"    r₁ = {r1s}",
                f"    r₂ = {r2s}",
                "",
                "  Circuito RLC: SOBRE-AMORTIGUADO.",
                "  La carga q(t) decae sin oscilar (R grande).", "",
                "  Solución general:",
                "",
                "  ┌─────────────────────────────────────────────────────────┐",
                f"  │   y(t) = C₁·e^({r1s}·t)  +  C₂·e^({r2s}·t)           │",
                "  └─────────────────────────────────────────────────────────┘", "",
            ]
        elif df == 0:
            cs = "rep"
            r1n = r2n = float(r1s)
            alp = bet = None
            L += [
                "  Δ = 0   →   CASO 2: Raíz real repetida",
                "",
                f"    r₁ = r₂ = {r1s}",
                "",
                "  Circuito RLC: AMORTIGUAMIENTO CRÍTICO.",
                "  La carga regresa al equilibrio en el menor tiempo sin oscilar.", "",
                "  Solución general:",
                "",
                "  ┌─────────────────────────────────────────────────────────┐",
                f"  │   y(t) = ( C₁  +  C₂·t ) · e^({r1s}·t)               │",
                "  └─────────────────────────────────────────────────────────┘", "",
            ]
        else:
            cs  = "comp"
            als = sp.simplify(-b / (2*a))
            bes = sp.simplify(sp.sqrt(-disc) / (2*a))
            alp, bet = float(als), float(bes)
            r1n = complex(alp,  bet)
            r2n = complex(alp, -bet)
            L += [
                f"  Δ = {disc} < 0   →   CASO 3: Raíces complejas conjugadas",
                "",
                "  Las raíces tienen la forma  r = α ± βi",
                "",
                f"    α (parte real)        = {als}",
                f"    β (parte imaginaria)  = {bes}",
                "",
                f"    r₁ = {als} + {bes}·i",
                f"    r₂ = {als} − {bes}·i",
                "",
                "  Aplicando la fórmula de Euler:",
                f"    y₁ = e^({als}·t)·cos({bes}·t)     y₂ = e^({als}·t)·sin({bes}·t)",
                "",
                "  Circuito RLC: SUB-AMORTIGUADO.",
                f"  La carga oscila con frecuencia {bes}, amplitud decae como e^({als}·t).", "",
                "  Solución general:",
                "",
                "  ┌─────────────────────────────────────────────────────────┐",
                f"  │   y(t) = e^({als}·t)·[ C₁·cos({bes}·t) + C₂·sin({bes}·t) ]   │",
                "  └─────────────────────────────────────────────────────────┘", "",
            ]

        L += [
            "  C₁ y C₂ se determinan con las condiciones iniciales:",
            "    y(0)  = carga/posición inicial",
            "    y′(0) = corriente/velocidad inicial",
            "",
            "=" * 62,
        ]

        self._write_out(L)
        # CORRECCIÓN: mostrar solución primero, gráfica en segundo plano
        self._plot_t2(cs, r1n, r2n, alp, bet)

    # ─────────────────────────────────────────────────────────────────────────
    #  GRÁFICAS
    # ─────────────────────────────────────────────────────────────────────────
    def _sax(self, ax, title, xl="x", yl="y"):
        ax.set_facecolor("#0d1117")
        ax.set_title(title, color=TEXT, fontsize=12, pad=12, fontweight="bold")
        ax.set_xlabel(xl, color=DIM, fontsize=11)
        ax.set_ylabel(yl, color=DIM, fontsize=11)
        ax.tick_params(colors=DIM, labelsize=9)
        ax.axhline(0, color=BDR, lw=0.9)
        ax.axvline(0, color=BDR, lw=0.9)
        for s in ax.spines.values(): s.set_color(BDR)

    def _blank_graph(self):
        self._fig.clear(); self._fig.patch.set_facecolor(INP)
        ax = self._fig.add_subplot(111); ax.set_facecolor("#0d1117")
        ax.text(0.5, 0.5, "Resuelve una ecuación\npara ver la gráfica aquí",
                ha="center", va="center", color=DIM, fontsize=14,
                transform=ax.transAxes, style="italic", multialignment="center")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(BDR)
        self._cv.draw()

    def _plot_t1(self, M, N):
        self._fig.clear(); self._fig.patch.set_facecolor(INP)
        ax = self._fig.add_subplot(111)
        self._sax(ax, "Campo de direcciones  —  dy/dx = −M(x,y) / N(x,y)")
        xv = np.linspace(-4, 4, 22); yv = np.linspace(-4, 4, 22)
        Xg, Yg = np.meshgrid(xv, yv)
        try:
            Mf = sp.lambdify((xs, ys), M, "numpy")
            Nf = sp.lambdify((xs, ys), N, "numpy")
            with np.errstate(all="ignore"):
                Mv = np.array(Mf(Xg, Yg), dtype=float)
                Nv = np.array(Nf(Xg, Yg), dtype=float)
                U  = np.where(np.isfinite(Nv),  Nv, 0.)
                V  = np.where(np.isfinite(Mv), -Mv, 0.)
                nm = np.sqrt(U**2 + V**2); nm[nm == 0] = 1
                U /= nm; V /= nm
                mask = np.isfinite(U) & np.isfinite(V)
            ax.quiver(Xg[mask], Yg[mask], U[mask], V[mask],
                       color=CYAN, alpha=0.72, scale=28,
                       width=0.003, headwidth=4, headlength=5)
            ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        except Exception as e:
            ax.text(0.5, 0.5, f"No se pudo graficar:\n{e}",
                     ha="center", va="center", color=RED, transform=ax.transAxes)
        self._fig.tight_layout(pad=2.0)
        self._cv.draw()
        # CORRECCIÓN: NO cambiar a pestaña de gráfica automáticamente

    def _plot_t2(self, cs, r1, r2, alp, bet):
        self._fig.clear(); self._fig.patch.set_facecolor(INP)
        ax = self._fig.add_subplot(111)
        titles = {
            "dist": "Caso 1: Sobre-amortiguado — Raíces reales distintas",
            "rep":  "Caso 2: Críticamente amortiguado — Raíz repetida",
            "comp": "Caso 3: Sub-amortiguado — Raíces complejas (oscila)",
        }
        self._sax(ax, f"Respuesta transitoria y(t)  ·  {titles[cs]}",
                  xl="t  (tiempo)", yl="y(t)  (señal)")
        xp  = np.linspace(0, 8, 800)
        ics = [(1, 0), (0, 1), (1, 0.5), (1, -0.5), (0.5, 0.5)]
        for i, (c1, c2) in enumerate(ics):
            with np.errstate(all="ignore"):
                if cs == "dist":
                    r1r = r1.real if isinstance(r1, complex) else float(r1)
                    r2r = r2.real if isinstance(r2, complex) else float(r2)
                    yp  = c1*np.exp(r1r*xp) + c2*np.exp(r2r*xp)
                elif cs == "rep":
                    r1r = r1.real if isinstance(r1, complex) else float(r1)
                    yp  = (c1 + c2*xp) * np.exp(r1r*xp)
                else:
                    yp  = np.exp(alp*xp)*(c1*np.cos(bet*xp)+c2*np.sin(bet*xp))
            mask = np.isfinite(yp) & (np.abs(yp) < 15)
            if mask.any():
                ax.plot(xp[mask], yp[mask], color=PAL[i], lw=1.9, alpha=0.92,
                         label=f"y(0)={c1},  y′(0)={c2}")
        ax.set_xlim(0, 8); ax.set_ylim(-6, 6)
        ax.axhline(0, color=BDR, lw=1.2, ls="--", alpha=0.6)
        ax.legend(fontsize=9, facecolor="#161b22", labelcolor=TEXT,
                   edgecolor=BDR, loc="best", framealpha=0.92)
        self._fig.tight_layout(pad=2.0)
        self._cv.draw()
        # CORRECCIÓN: NO cambiar a pestaña de gráfica automáticamente

    # ─────────────────────────────────────────────────────────────────────────
    #  FORMATO
    # ─────────────────────────────────────────────────────────────────────────
    def _fmt_eq(self, a, b, c):
        parts = []
        for val, term in [(a, "y″"), (b, "y′"), (c, "y")]:
            if val == 0: continue
            if not parts:
                if val == 1:    parts.append(term)
                elif val == -1: parts.append(f"−{term}")
                else:           parts.append(f"{val:g}{term}")
            else:
                if val == 1:    parts.append(f"+ {term}")
                elif val == -1: parts.append(f"− {term}")
                elif val > 0:   parts.append(f"+ {val:g}{term}")
                else:           parts.append(f"− {abs(val):g}{term}")
        return " ".join(parts) or "0"

    def _fmt_char(self, a, b, c):
        parts = []
        for val, term in [(a, "r²"), (b, "r"), (c, "")]:
            if val == 0: continue
            d = f"{val:g}{term}" if term else f"{val:g}"
            if not parts: parts.append(d)
            elif val > 0: parts.append(f"+ {d}")
            else:         parts.append(f"− {abs(val):g}{term}")
        return " ".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    #  E/S
    # ─────────────────────────────────────────────────────────────────────────
    def _write_out(self, lines: list):
        self._out.configure(state="normal")
        self._out.delete("1.0", "end")
        for line in lines:
            self._out.insert("end", line + "\n")
        self._out.see("1.0")
        self._out.configure(state="disabled")
        # CORRECCIÓN: siempre ir a Solución Paso a Paso primero
        self._tabs.set("  Solución Paso a Paso  ")
        self._status.configure(text="✓  Solución calculada. Ve a 'Gráfica de Solución' para ver la gráfica.",
                                text_color=GREEN)

    def _write_error(self, msg: str):
        self._out.configure(state="normal")
        self._out.delete("1.0", "end")
        self._out.insert("end", f"\n  ERROR: {msg}\n")
        self._out.configure(state="disabled")
        self._tabs.set("  Solución Paso a Paso  ")
        self._status.configure(text="Error — revisa los datos.", text_color=RED)

    # ─────────────────────────────────────────────────────────────────────────
    #  LIMPIAR
    # ─────────────────────────────────────────────────────────────────────────
    def _clear_all(self):
        self._show_welcome()
        self._blank_graph()
        self._status.configure(text="")
        if hasattr(self, "_fm"): self._fm.clear(); self._fn.clear()
        if hasattr(self, "_t2f"):
            for k in ("a","b","c"):
                if k in self._t2f:
                    self._t2f[k].clear()
        self._preview()

    def _show_welcome(self):
        HL = "─" * 62
        lines = [
            "=" * 62,
            "  Solucionador de Ecuaciones Diferenciales Homogéneas",
            "  Ingeniería en Sistemas Computacionales",
            "=" * 62,
            "",
            "  Instrucciones de uso:",
            "",
            "  1.  Selecciona el tipo de ecuación (panel izquierdo).",
            "  2.  Revisa el problema de aplicación en la pestaña",
            "      'Problema de Aplicación'.",
            "  3.  Haz clic en el campo que deseas llenar.",
            "  4.  Usa el teclado matemático o el teclado físico.",
            "  5.  Presiona '▶ Resolver paso a paso'.",
            "  6.  Primero verás la SOLUCIÓN en esta pestaña.",
            "  7.  Ve a 'Gráfica de Solución' para ver la gráfica.",
            "",
            HL,
            "",
            "  TIPO 1 — M(x,y)dx + N(x,y)dy = 0",
            "  Método: sustitución y = vx → separación de variables",
            "",
            "  TIPO 2 — ay″ + by′ + cy = 0",
            "  Método: ecuación característica ar² + br + c = 0",
            "    • Δ > 0  →  Caso 1: raíces reales distintas",
            "    • Δ = 0  →  Caso 2: raíz real repetida",
            "    • Δ < 0  →  Caso 3: raíces complejas conjugadas",
            "",
            "=" * 62,
        ]
        self._out.configure(state="normal")
        self._out.delete("1.0", "end")
        for line in lines:
            self._out.insert("end", line + "\n")
        self._out.see("1.0")
        self._out.configure(state="disabled")


# ─── Punto de entrada ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()