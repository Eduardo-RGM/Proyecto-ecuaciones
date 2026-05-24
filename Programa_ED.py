import warnings; warnings.filterwarnings("ignore")
import customtkinter as ctk
import sympy as sp
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import threading, ctypes, time

xs = sp.Symbol("x", real=True)
ys = sp.Symbol("y", real=True)
vs = sp.Symbol("v", real=True)
ts = sp.Symbol("t", positive=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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
PAL    = ["#388bfd","#3fb950","#ffa657","#bc8cff","#79c0ff","#f85149"]

VARS_T1 = [
    ("x",   "x",       "x",   "var"),
    ("y",   "y",       "y",   "var"),
    ("x²",  "x**2",    "x²",  "var"),
    ("y²",  "y**2",    "y²",  "var"),
    ("xy",  "x*y",     "xy",  "var"),
    ("x³",  "x**3",    "x³",  "var"),
    ("x²y", "x**2*y",  "x²y", "var"),
    ("xy²", "x*y**2",  "xy²", "var"),
    ("y³",  "y**3",    "y³",  "var"),
    ("(",   "(",        "(",   "pop"),
    (")",   ")",        ")",   "pcl"),
    ("/",   "/",        "÷",   "op"),
]

def _sp2h(expr) -> str:
    if expr is None: return "?"
    try:
        txt = str(sp.collect(sp.expand(expr), [xs, ys, vs]))
    except Exception:
        txt = str(expr)
    for old, new in [("**2","²"),("**3","³"),("**4","⁴"),("**5","⁵"),
                     ("sqrt(","√("),("log(","ln("),("*","·"),(" I","i"),(" - ","-")]:
        txt = txt.replace(old, new)
    txt = txt.replace("1·x","x").replace("1·y","y").replace("1·v","v")
    return txt



def _kill_thread(t):
    """Intenta forzar la terminación de un hilo de Python (best-effort)."""
    try:
        tid = t.ident
        if tid is None: return
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), ctypes.py_object(SystemExit))
    except Exception:
        pass

def _try_integrate(expr, var, method, timeout_sec):
    """Ejecuta una estrategia de integración con timeout. Devuelve (resultado, ok)."""
    result = [None]; done = [False]
    def worker():
        try:
            if method == "basic":
                result[0] = sp.integrate(expr, var)
            elif method == "apart":
                result[0] = sp.integrate(sp.apart(expr, var), var)
            elif method == "heurisch":
                result[0] = sp.heurisch(expr, var, rewrite=True)
            elif method == "meijerg":
                result[0] = sp.integrate(expr, var, meijerg=True)
        except Exception:
            pass
        finally:
            done[0] = True
    t = threading.Thread(target=worker, daemon=True)
    t.start(); t.join(timeout_sec)
    if not done[0]:
        _kill_thread(t)
    r = result[0]
    if r is None: return None, False
    if isinstance(r, sp.Integral): return None, False
    # Verificar que no contiene integrales sin resolver
    if r.has(sp.Integral): return None, False
    return r, True

def _integrate_safe(expr, var, timeout_sec=6):
    """
    Intenta integrar con varias estrategias en cascada.
    Primero simplifica/cancela la expresión para aligerar el trabajo simbólico.
    Retorna (resultado, ok).
    """
    try:
        expr = sp.cancel(expr)          # reduce fracciones algebraicas
        expr = sp.radsimp(expr)         # racionaliza radicales si los hay
    except Exception:
        pass

    # Estrategias por orden de velocidad
    strategies = [
        ("basic",    2),   # integración directa rápida
        ("apart",    3),   # fracciones parciales → integración
        ("heurisch", 4),   # heurística de Risch (suele resolver polinomios/racionales)
        ("meijerg",  5),   # funciones G de Meijer (último recurso, lento)
    ]
    for method, t in strategies:
        r, ok = _try_integrate(expr, var, method, min(t, timeout_sec))
        if ok:
            return r, True
        timeout_sec -= t
        if timeout_sec <= 0:
            break
    return None, False

# ─────────────────────────────────────────────────────────────────────────────

def _needs_mul(pk, nk):
    return ((nk in ("var","pop")) and pk in ("d","var","pcl")) or \
           ((nk == "d") and pk in ("var","pcl"))

def build_tokens(raw):
    result = []
    for internal, display, kind in raw:
        if result:
            lk = result[-1][2]
            if kind == "d" and lk == "d":
                result[-1] = (result[-1][0]+internal, result[-1][1]+display, "d")
                continue
            if _needs_mul(lk, kind):
                result.append(("*","·","_m"))
        result.append((internal, display, kind))
    return result


class ExprField(ctk.CTkFrame):
    def __init__(self, master, label="", placeholder="Clic aquí → usa el teclado", **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self._tok = []
        self._ph  = placeholder

        if label:
            self._lbl_title = ctk.CTkLabel(
                self, text=label,
                font=ctk.CTkFont("Consolas", 13, "bold"),
                text_color=CYAN, cursor="hand2")
            self._lbl_title.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 2))
            self._bind_click(self._lbl_title)

        self._box = ctk.CTkFrame(
            self, fg_color=INP, corner_radius=8,
            border_width=2, border_color=BDR, cursor="hand2")
        self._box.grid(row=1, column=0, sticky="ew")
        self._box.grid_columnconfigure(0, weight=1)

        self._lbl = ctk.CTkLabel(
            self._box, text="",
            font=ctk.CTkFont("Consolas", 15),
            text_color=TEXT, anchor="w", cursor="hand2")
        self._lbl.grid(row=0, column=0, sticky="ew", padx=12, pady=11)

        self._cur = ctk.CTkLabel(
            self._box, text="",
            font=ctk.CTkFont("Consolas", 18, "bold"),
            text_color=ACC, cursor="hand2")
        self._cur.grid(row=0, column=1, padx=(0, 10))

        self._refresh()
        self.after(50,  self._install_bindings)
        self.after(300, self._install_bindings)

    def _install_bindings(self):
        self._bind_click_recursive(self._box)

    def _bind_click_recursive(self, widget):
        self._bind_click(widget)
        for child in widget.winfo_children():
            self._bind_click_recursive(child)

    def _bind_click(self, widget):
        try: widget.bind("<Button-1>", self._on_click, add="+")
        except Exception: pass

    def _on_click(self, e):
        self.event_generate("<<FieldClick>>")

    def activate(self, on):
        self._box.configure(border_color=BACT if on else BDR)
        self._cur.configure(text="│" if on else "")

    def push(self, internal, display, kind):
        t = self._tok
        if t:
            lk = t[-1][2]
            if kind == "d" and lk == "d":
                t[-1] = (t[-1][0]+internal, t[-1][1]+display, "d")
                self._refresh(); return
            if _needs_mul(lk, kind):
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

    def clear(self):       self._tok.clear(); self._refresh()

    def negate(self):
        if self._tok and self._tok[0][2] == "_neg":
            self._tok.pop(0)
        else:
            self._tok.insert(0, ("-","−","_neg"))
        self._refresh()

    def set_tokens(self, raw):
        self._tok = build_tokens(raw); self._refresh()

    def get_expr(self): return "".join(t[0] for t in self._tok)
    def get_disp(self): return "".join(t[1] for t in self._tok)

    def _refresh(self):
        d = self.get_disp()
        self._lbl.configure(
            text=d if d else self._ph,
            text_color=TEXT if d else DIM)


class EqPreview(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=CARD, corner_radius=10,
                          border_width=1, border_color=BDR, **kw)
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="  Vista previa", font=ctk.CTkFont("Segoe UI",10),
                     text_color=DIM).grid(row=0,column=0,padx=12,pady=(8,2),sticky="w")
        self._lbl = ctk.CTkLabel(self, text=" — ",
            font=ctk.CTkFont("Consolas",14), text_color=GREEN,
            wraplength=440, justify="left")
        self._lbl.grid(row=1,column=0,padx=12,pady=(2,10),sticky="ew")

    def set_t1(self, m, n):
        m = m if m else "M(x,y)"; n = n if n else "N(x,y)"
        self._lbl.configure(text=f"  ({m}) dx  +  ({n}) dy  =  0")

    def set_t2(self, a, b, c):
        parts = []
        for v,t in [(a,"y″"),(b,"y′"),(c,"y")]:
            if not v: continue
            if not parts:           parts.append(f"{v}{t}")
            elif v.startswith("−"): parts.append(f" − {v[1:]}{t}")
            else:                   parts.append(f" + {v}{t}")
        self._lbl.configure(text=f"  {''.join(parts) or '0'}  =  0")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ED Homogéneas")
        self.geometry("1460x900")
        self.minsize(1220,800)
        self.configure(fg_color=BG)
        self._active_field = None
        self._eq_type = "tipo2"
        self._solving  = False   # bandera anti-doble-click
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        lo = ctk.CTkFrame(self, width=510, corner_radius=0, fg_color=SURF)
        lo.grid(row=0,column=0,sticky="nsew"); lo.grid_propagate(False)
        lo.grid_rowconfigure(0,weight=1); lo.grid_columnconfigure(0,weight=1)
        self._lscroll = ctk.CTkScrollableFrame(lo, fg_color="transparent",
            scrollbar_button_color=BDR, scrollbar_button_hover_color=ACC)
        self._lscroll.grid(row=0,column=0,sticky="nsew")
        self._lscroll.grid_columnconfigure(0,weight=1)
        self._build_left(self._lscroll)
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0,column=1,sticky="nsew",padx=22,pady=22)
        right.grid_columnconfigure(0,weight=1); right.grid_rowconfigure(1,weight=1)
        self._build_right(right)

    def _build_left(self, p):
        r = 0
        hdr = ctk.CTkFrame(p, fg_color=CARD, corner_radius=12, border_width=1, border_color=BDR)
        hdr.grid(row=r,column=0,padx=16,pady=(18,10),sticky="ew"); r+=1
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="ED Homogéneas",
            font=ctk.CTkFont("Segoe UI",22,"bold"), text_color=TEXT
            ).grid(row=0,column=0,padx=16,pady=(14,14),sticky="w")
        ctk.CTkLabel(p, text="  Tipo de ecuación",
            font=ctk.CTkFont("Segoe UI",11,"bold"), text_color=DIM
            ).grid(row=r,column=0,padx=16,pady=(4,4),sticky="w"); r+=1
        seg = ctk.CTkSegmentedButton(p, values=["1° Orden","2° Orden"],
            command=self._on_type_change,
            font=ctk.CTkFont("Segoe UI",12,"bold"), fg_color=CARD,
            selected_color=ACC, selected_hover_color="#2879f0",
            unselected_color=CARD, unselected_hover_color=BDR,
            text_color=TEXT, corner_radius=8, height=42)
        seg.set("2° Orden")
        seg.grid(row=r,column=0,padx=16,pady=(0,8),sticky="ew"); r+=1
        self._type_sub = ctk.CTkLabel(p, text="  ay″ + by′ + cy = 0",
            font=ctk.CTkFont("Consolas",11), text_color=CYAN)
        self._type_sub.grid(row=r,column=0,padx=16,pady=(0,4),sticky="w"); r+=1
        self._hsep(p,r); r+=1
        self._inp = ctk.CTkFrame(p, fg_color="transparent")
        self._inp.grid(row=r,column=0,padx=16,pady=4,sticky="ew"); r+=1
        self._inp.grid_columnconfigure(0,weight=1)
        self._hsep(p,r); r+=1
        self._prev = EqPreview(p)
        self._prev.grid(row=r,column=0,padx=16,pady=(0,6),sticky="ew"); r+=1
        self._hsep(p,r); r+=1
        ctk.CTkLabel(p, text="  Teclado matemático",
            font=ctk.CTkFont("Segoe UI",11,"bold"), text_color=DIM
            ).grid(row=r,column=0,padx=16,pady=(4,4),sticky="w"); r+=1
        self._kbd = ctk.CTkFrame(p, fg_color=CARD, corner_radius=12, border_width=1, border_color=BDR)
        self._kbd.grid(row=r,column=0,padx=16,pady=(0,6),sticky="ew"); r+=1
        self._kbd.grid_columnconfigure(0,weight=1)
        self._hsep(p,r); r+=1
        self._btn_solve = ctk.CTkButton(p, text="▶  Resolver paso a paso",
            font=ctk.CTkFont("Segoe UI",14,"bold"), height=52, corner_radius=10,
            fg_color=ACC, hover_color="#2879f0", command=self._solve)
        self._btn_solve.grid(row=r,column=0,padx=16,pady=(8,4),sticky="ew"); r+=1
        ctk.CTkButton(p, text="Limpiar todo",
            font=ctk.CTkFont("Segoe UI",11), height=36, corner_radius=8,
            fg_color="transparent", border_width=1, border_color=BDR,
            hover_color=CARD, text_color=DIM, command=self._clear_all
            ).grid(row=r,column=0,padx=16,pady=(0,6),sticky="ew"); r+=1
        self._status = ctk.CTkLabel(p, text="",
            font=ctk.CTkFont("Segoe UI",10), text_color=DIM, wraplength=474)
        self._status.grid(row=r,column=0,padx=16,pady=(0,20),sticky="w"); r+=1
        self._build_t2_inputs()
        self._build_keyboard()

    def _hsep(self, p, row):
        ctk.CTkFrame(p, height=1, fg_color=BDR).grid(row=row, column=0, padx=16, pady=3, sticky="ew")

    # ─── TIPO 1 ──────────────────────────────────────────────────────────────
    def _build_t1_inputs(self):
        self._clear_inp_area()
        f = self._inp
        f.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(f, text="  M(x, y) · dx  +  N(x, y) · dy  =  0",
            font=ctk.CTkFont("Consolas",12,"bold"), text_color=CYAN
            ).grid(row=0, column=0, sticky="w", pady=(4,2))

        instr = ctk.CTkFrame(f, fg_color="#0d2a1e", corner_radius=8, border_width=1, border_color="#1a4a2e")
        instr.grid(row=1, column=0, sticky="ew", pady=(0,8))
        instr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(instr,
            text="  ① Haz clic en la tarjeta M o N para activarla (se resalta en azul)\n"
                 "  ② Usa los botones de variables (azul) o ingresa coeficientes con el teclado físico\n"
                 "  ③ Teclado físico: dígitos, punto, ± (guión), ⌫ Backspace · Tab = siguiente campo",
            font=ctk.CTkFont("Segoe UI",10), text_color=GREEN, justify="left"
            ).grid(row=0, column=0, padx=8, pady=6, sticky="w")

        mfr = ctk.CTkFrame(f, fg_color=CARD, corner_radius=10, border_width=2, border_color=BDR)
        mfr.grid(row=2, column=0, sticky="ew", pady=(0,6))
        mfr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(mfr, text="M", font=ctk.CTkFont("Consolas",24,"bold"), text_color=CYAN
            ).grid(row=0, column=0, pady=(10,0))
        ctk.CTkLabel(mfr, text="coeficiente de dx", font=ctk.CTkFont("Segoe UI",9), text_color=DIM
            ).grid(row=1, column=0, pady=(0,4))
        self._fm = ExprField(mfr, placeholder="ej: x²+y²")
        self._fm.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,10))
        self._fm.bind("<<FieldClick>>", lambda e: self._activate_t1(self._fm, mfr))
        mfr.bind("<Button-1>", lambda e: self._activate_t1(self._fm, mfr))
        self._t1_mfr = mfr

        nfr = ctk.CTkFrame(f, fg_color=CARD, corner_radius=10, border_width=2, border_color=BDR)
        nfr.grid(row=3, column=0, sticky="ew", pady=(0,10))
        nfr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(nfr, text="N", font=ctk.CTkFont("Consolas",24,"bold"), text_color=PURPLE
            ).grid(row=0, column=0, pady=(10,0))
        ctk.CTkLabel(nfr, text="coeficiente de dy", font=ctk.CTkFont("Segoe UI",9), text_color=DIM
            ).grid(row=1, column=0, pady=(0,4))
        self._fn = ExprField(nfr, placeholder="ej: −2xy")
        self._fn.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,10))
        self._fn.bind("<<FieldClick>>", lambda e: self._activate_t1(self._fn, nfr))
        nfr.bind("<Button-1>", lambda e: self._activate_t1(self._fn, nfr))
        self._t1_nfr = nfr

        self.bind("<Key>", self._on_key_press)

        ctk.CTkLabel(f, text="  Ejemplos predefinidos:",
            font=ctk.CTkFont("Segoe UI",10), text_color=DIM
            ).grid(row=4, column=0, sticky="w", pady=(0,4))

        examples = [
            ("M = x²+y²    N = −2xy      ← caso CPU (parábolas)",
             [("x**2","x²","var"),("+","+","op"),("y**2","y²","var")],
             [("-","−","_neg"),("2","2","d"),("x","x","var"),("y","y","var")]),
            ("M = 2xy       N = y²−x²    (círculos)",
             [("2","2","d"),("x","x","var"),("y","y","var")],
             [("y**2","y²","var"),("-","-","op"),("x**2","x²","var")]),
            ("M = x³+y³   N = x²y",
             [("x**3","x³","var"),("+","+","op"),("y**3","y³","var")],
             [("x**2*y","x²y","var")]),
            ("M = x²+xy   N = y²",
             [("x**2","x²","var"),("+","+","op"),("x","x","var"),("y","y","var")],
             [("y**2","y²","var")]),
        ]
        for i,(lbl,mt,nt) in enumerate(examples):
            ctk.CTkButton(f, text=lbl, height=32, corner_radius=6,
                fg_color=CARD, hover_color=BDR,
                font=ctk.CTkFont("Consolas",11), text_color=DIM,
                border_width=1, border_color=BDR,
                command=lambda m=mt,n=nt: self._load_t1(m,n)
                ).grid(row=5+i, column=0, pady=2, sticky="ew")

        self.after(150, lambda: self._activate_t1(self._fm, mfr))
        self._prev.set_t1("","")

    def _activate_t1(self, ef, fr):
        if self._active_field and self._active_field is not ef:
            self._active_field.activate(False)
            for fr_attr in ("_t1_mfr", "_t1_nfr"):
                if hasattr(self, fr_attr):
                    getattr(self, fr_attr).configure(border_color=BDR)
        self._active_field = ef
        ef.activate(True)
        fr.configure(border_color=BACT)

    def _load_t1(self, mt, nt):
        self._fm.set_tokens(mt)
        self._fn.set_tokens(nt)
        self._preview()

    # ─── TIPO 2 ──────────────────────────────────────────────────────────────
    def _build_t2_inputs(self):
        self._clear_inp_area()
        f = self._inp
        f.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f, text="  a · y″  +  b · y′  +  c · y  =  0",
            font=ctk.CTkFont("Consolas",12,"bold"), text_color=CYAN
            ).grid(row=0,column=0,sticky="w",pady=(4,2))
        instr = ctk.CTkFrame(f, fg_color="#0d2a1e", corner_radius=8, border_width=1, border_color="#1a4a2e")
        instr.grid(row=1,column=0,sticky="ew",pady=(0,8))
        instr.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(instr,
            text="  ① Haz clic en la tarjeta a, b o c  (se resalta en azul)\n"
                 "  ② Escribe el valor con teclado físico o el de pantalla\n"
                 "  ③ Acepta enteros o decimales (ej: 1, −2, 0.5)\n"
                 "  ④ Usa Tab para pasar al siguiente coeficiente",
            font=ctk.CTkFont("Segoe UI",10), text_color=GREEN, justify="left"
            ).grid(row=0,column=0,padx=8,pady=6,sticky="w")
        cf = ctk.CTkFrame(f, fg_color="transparent")
        cf.grid(row=2,column=0,sticky="ew",pady=(0,10))
        cf.grid_columnconfigure((0,1,2),weight=1)
        self._t2f = {}
        for col,(key,deriv,color,sub) in enumerate([
                ("a","y″",ACC,"(2do orden)"),
                ("b","y′",PURPLE,"(1er orden)"),
                ("c","y",GREEN,"(término ind.)")]):
            fr = ctk.CTkFrame(cf, fg_color=CARD, corner_radius=10, border_width=2, border_color=BDR)
            fr.grid(row=0,column=col,padx=4,sticky="nsew")
            fr.grid_columnconfigure(0,weight=1)
            ctk.CTkLabel(fr,text=key,font=ctk.CTkFont("Consolas",24,"bold"),text_color=color
                ).grid(row=0,column=0,pady=(10,0))
            ctk.CTkLabel(fr,text=f"de {deriv}",font=ctk.CTkFont("Segoe UI",9),text_color=DIM
                ).grid(row=1,column=0,pady=(0,2))
            ctk.CTkLabel(fr,text=sub,font=ctk.CTkFont("Segoe UI",8),text_color=DIM
                ).grid(row=2,column=0,pady=(0,4))
            ef = ExprField(fr, placeholder="0")
            ef.grid(row=3,column=0,sticky="ew",padx=6,pady=(0,10))
            ef.bind("<<FieldClick>>", lambda e,ef=ef,fr=fr: self._activate_t2(ef,fr))
            fr.bind("<Button-1>",     lambda e,ef=ef,fr=fr: self._activate_t2(ef,fr))
            self._t2f[key] = ef
            self._t2f[f"_{key}_fr"] = fr
        self.bind("<Key>", self._on_key_press)
        ctk.CTkLabel(f,text="  Ejemplos predefinidos:",
            font=ctk.CTkFont("Segoe UI",10),text_color=DIM
            ).grid(row=3,column=0,sticky="w",pady=(0,4))
        for i,(lbl,a,b,c) in enumerate([
                ("y″ − 5y′ + 6y = 0     (raíces reales distintas)",1,-5,6),
                ("y″ − 6y′ + 9y = 0     (raíz repetida)",          1,-6,9),
                ("y″ + 4y = 0            (complejas puras)",         1, 0,4),
                ("y″ + 2y′ + 5y = 0     (sub-amortiguado)",         1, 2,5),
                ("y″ + 2y′ + 4y = 0     ← Caso RLC",               1, 2,4)]):
            ctk.CTkButton(f,text=lbl,height=32,corner_radius=6,
                fg_color=CARD,hover_color=BDR,font=ctk.CTkFont("Consolas",11),text_color=DIM,
                border_width=1,border_color=BDR,command=lambda a=a,b=b,c=c:self._load_t2(a,b,c)
                ).grid(row=4+i,column=0,pady=2,sticky="ew")
        self.after(150, lambda: self._activate_t2(self._t2f["a"], self._t2f["_a_fr"]))
        self._prev.set_t2("","","")

    def _activate_t2(self, ef, fr):
        if self._active_field and self._active_field is not ef:
            self._active_field.activate(False)
            for k in "abc":
                fk = f"_{k}_fr"
                if fk in self._t2f: self._t2f[fk].configure(border_color=BDR)
        self._active_field = ef
        ef.activate(True)
        fr.configure(border_color=BACT)

    # ─── Teclado físico ──────────────────────────────────────────────────────
    def _on_key_press(self, event):
        if not self._active_field: return
        ch, sym = event.char, event.keysym

        if sym == "BackSpace":
            self._active_field.backspace(); self._preview(); return
        if sym in ("minus","KP_Subtract") or ch == "-":
            self._active_field.negate(); self._preview(); return
        if ch.isdigit():
            self._active_field.push(ch, ch, "d"); self._preview(); return
        if ch == ".":
            self._active_field.push(".", ".", "d"); self._preview(); return
        if sym == "Tab":
            if self._eq_type == "tipo2":
                order = ["a","b","c"]
                cur = next((k for k in order if self._t2f.get(k) is self._active_field), None)
                if cur:
                    nxt = order[(order.index(cur)+1) % 3]
                    self._activate_t2(self._t2f[nxt], self._t2f[f"_{nxt}_fr"])
            else:
                if self._active_field is self._fm:
                    self._activate_t1(self._fn, self._t1_nfr)
                elif self._active_field is self._fn:
                    self._activate_t1(self._fm, self._t1_mfr)

    def _load_t2(self, a, b, c):
        for key,val in [("a",a),("b",b),("c",c)]:
            toks=[]
            if val < 0:
                sv=str(abs(int(val))) if val==int(val) else str(abs(val))
                toks=[("-","−","_neg"),(sv,sv,"d")]
            elif val != 0:
                sv=str(int(val)) if val==int(val) else str(val)
                toks=[(sv,sv,"d")]
            self._t2f[key].set_tokens(toks)
        self._preview()

    def _clear_inp_area(self):
        for w in self._inp.winfo_children(): w.destroy()
        self._active_field = None
        # Eliminar referencias a campos anteriores para evitar errores en _clear_all
        for attr in ("_fm","_fn","_t1_mfr","_t1_nfr","_t2f"):
            if hasattr(self, attr):
                delattr(self, attr)
        try: self.unbind("<Key>")
        except Exception: pass

    def _preview(self):
        if self._eq_type == "tipo1":
            m = self._fm.get_disp() if hasattr(self,"_fm") else ""
            n = self._fn.get_disp() if hasattr(self,"_fn") else ""
            self._prev.set_t1(m,n)
        else:
            vals = {k:self._t2f[k].get_disp() for k in "abc"} if hasattr(self,"_t2f") else {}
            self._prev.set_t2(vals.get("a",""),vals.get("b",""),vals.get("c",""))

    # ─── Teclado de pantalla ─────────────────────────────────────────────────
    def _build_keyboard(self):
        for w in self._kbd.winfo_children(): w.destroy()
        k = self._kbd
        k.grid_columnconfigure(0, weight=1)
        row = 0

        def mkbtn(parent,txt,cmd,r,c,span=1,h=42,fg=CARD,hv=BDR,tc=TEXT,bold=False):
            ctk.CTkButton(parent,text=txt,height=h,
                font=ctk.CTkFont("Consolas",13,"bold" if bold else "normal"),
                corner_radius=7,fg_color=fg,hover_color=hv,text_color=tc,command=cmd
                ).grid(row=r,column=c,columnspan=span,padx=3,pady=3,sticky="ew")

        if self._eq_type == "tipo1":
            lf = ctk.CTkFrame(k, fg_color="transparent")
            lf.grid(row=row,column=0,padx=10,pady=(12,4),sticky="ew"); row+=1
            ctk.CTkLabel(lf,text="  Variables y expresiones  (clic en M o N primero)",
                font=ctk.CTkFont("Segoe UI",10,"bold"),text_color=DIM).pack(anchor="w")

            vf = ctk.CTkFrame(k, fg_color="#0d2035", corner_radius=8)
            vf.grid(row=row,column=0,padx=10,pady=(0,4),sticky="ew"); row+=1
            for i in range(4): vf.grid_columnconfigure(i, weight=1)
            for idx,(lbl,intl,disp,kind) in enumerate(VARS_T1):
                r2,c2 = divmod(idx, 4)
                mkbtn(vf,lbl,lambda i=intl,d=disp,kk=kind:self._kpush(i,d,kk),
                      r2,c2,fg="#0d2035",hv="#1a3d5c",tc=CYAN,bold=True,h=40)

            opf = ctk.CTkFrame(k, fg_color="transparent")
            opf.grid(row=row,column=0,padx=10,pady=(0,4),sticky="ew"); row+=1
            opf.grid_columnconfigure((0,1),weight=1)
            ctk.CTkButton(opf,text="  +  Suma",height=40,corner_radius=7,
                font=ctk.CTkFont("Consolas",14,"bold"),
                fg_color="#0d2a1e",hover_color="#1a4a2e",text_color=GREEN,
                command=lambda:self._kpush("+","+","op")
                ).grid(row=0,column=0,padx=3,pady=3,sticky="ew")
            ctk.CTkButton(opf,text="  −  Resta",height=40,corner_radius=7,
                font=ctk.CTkFont("Consolas",14,"bold"),
                fg_color="#0d2a1e",hover_color="#1a4a2e",text_color=GREEN,
                command=lambda:self._kpush("-","-","op")
                ).grid(row=0,column=1,padx=3,pady=3,sticky="ew")

            ctk.CTkFrame(k,height=1,fg_color=BDR).grid(row=row,column=0,padx=10,pady=4,sticky="ew"); row+=1

        lf2 = ctk.CTkFrame(k, fg_color="transparent")
        lf2.grid(row=row,column=0,padx=10,pady=(6,4),sticky="ew"); row+=1
        hint = ("  Coeficientes numéricos" if self._eq_type=="tipo1"
                else "  Valor del coeficiente  (teclado físico también funciona · Tab=siguiente)")
        ctk.CTkLabel(lf2,text=hint,font=ctk.CTkFont("Segoe UI",10,"bold"),text_color=DIM).pack(anchor="w")

        nf = ctk.CTkFrame(k, fg_color="transparent")
        nf.grid(row=row,column=0,padx=10,pady=(0,12),sticky="ew"); row+=1
        for i in range(5): nf.grid_columnconfigure(i, weight=1)

        for ri,rowdata in enumerate([
            [("7","7","d"),("8","8","d"),("9","9","d"),("+","+","op"),("−","-","op")],
            [("4","4","d"),("5","5","d"),("6","6","d"),(".",".","d"), ("⌫","","back")],
            [("1","1","d"),("2","2","d"),("3","3","d"),("±","","neg"),("✕","","clr")],
            [("0","0","d"),],
        ]):
            for ci,(lbl,val,kind) in enumerate(rowdata):
                if   kind=="back": mkbtn(nf,"⌫  Borrar",  self._kback,ri,ci,fg="#2a1020",hv="#4a2030",tc="#ff9999",h=42)
                elif kind=="neg":  mkbtn(nf,"± signo",    self._kneg, ri,ci,fg=CARD,    hv=BDR,     tc=DIM,      h=42)
                elif kind=="clr":  mkbtn(nf,"✕  Limpiar", self._kclr, ri,ci,fg="#2a1020",hv="#4a2030",tc="#ff9999",h=42)
                elif kind=="op":   mkbtn(nf,lbl,lambda v=val,l=lbl:self._kpush(v,l,"op"),
                                        ri,ci,fg="#0d2a1e",hv="#1a4a2e",tc=GREEN,bold=True,h=42)
                else:
                    span = 5 if lbl=="0" else 1
                    mkbtn(nf,lbl,lambda v=val,l=lbl:self._kpush(v,l,"d"),ri,ci,span=span,h=42)

    def _kpush(self, i, d, k):
        if self._active_field:
            self._active_field.push(i, d, k); self._preview()
        else:
            self._status.configure(
                text="⚠  Haz clic primero en la tarjeta M o N para activarla.",
                text_color=YELLOW)
    def _kback(self):
        if self._active_field: self._active_field.backspace(); self._preview()
    def _kneg(self):
        if self._active_field: self._active_field.negate(); self._preview()
    def _kclr(self):
        if self._active_field: self._active_field.clear(); self._preview()

    def _on_type_change(self, val):
        if self._solving: return   # no cambiar mientras se resuelve
        self._eq_type = "tipo1" if "1" in val else "tipo2"
        if self._eq_type == "tipo1":
            self._type_sub.configure(text="  M(x,y)·dx + N(x,y)·dy = 0")
            self._build_t1_inputs()
        else:
            self._type_sub.configure(text="  ay″ + by′ + cy = 0")
            self._build_t2_inputs()
        self._build_keyboard()
        self._update_app_tab()

    # ─── Panel derecho ───────────────────────────────────────────────────────
    def _build_right(self, p):
        ctk.CTkLabel(p, text="Procedimiento",
            font=ctk.CTkFont("Segoe UI",19,"bold"), text_color=TEXT
            ).grid(row=0,column=0,sticky="w",pady=(0,14))
        self._tabs = ctk.CTkTabview(p, anchor="nw",
            segmented_button_fg_color=CARD,
            segmented_button_selected_color=ACC,
            segmented_button_selected_hover_color="#2879f0",
            fg_color=CARD, corner_radius=12)
        self._tabs.grid(row=1,column=0,sticky="nsew")
        for name in ["  Problema de Aplicación  ","  Solución Paso a Paso  ","  Gráfica de Solución  "]:
            t=self._tabs.add(name); t.grid_columnconfigure(0,weight=1); t.grid_rowconfigure(0,weight=1)
        self._tab_app  = self._tabs.tab("  Problema de Aplicación  ")
        self._tab_proc = self._tabs.tab("  Solución Paso a Paso  ")
        self._tab_graf = self._tabs.tab("  Gráfica de Solución  ")
        self._out = ctk.CTkTextbox(self._tab_proc,
            font=ctk.CTkFont("Consolas",13), wrap="word", corner_radius=8,
            fg_color=INP, text_color=TEXT,
            scrollbar_button_color=BDR, scrollbar_button_hover_color=ACC)
        self._out.grid(row=0,column=0,sticky="nsew",padx=2,pady=2)
        self._fig = Figure(figsize=(7,5), dpi=100)
        self._fig.patch.set_facecolor(INP)
        self._cv  = FigureCanvasTkAgg(self._fig, master=self._tab_graf)
        self._cv.get_tk_widget().configure(bg=INP)
        self._cv.get_tk_widget().grid(row=0,column=0,sticky="nsew")
        self._app_scroll = ctk.CTkScrollableFrame(self._tab_app,
            fg_color="transparent",
            scrollbar_button_color=BDR, scrollbar_button_hover_color=ACC)
        self._app_scroll.grid(row=0,column=0,sticky="nsew")
        self._app_scroll.grid_columnconfigure(0,weight=1)
        self._show_welcome(); self._blank_graph(); self._update_app_tab()

    def _update_app_tab(self):
        for w in self._app_scroll.winfo_children(): w.destroy()
        if self._eq_type=="tipo1": self._build_app_t1()
        else:                       self._build_app_t2()

    def _acard(self,parent,row):
        f=ctk.CTkFrame(parent,fg_color=CARD,corner_radius=10,border_width=1,border_color=BDR)
        f.grid(row=row,column=0,padx=8,pady=6,sticky="ew")
        f.grid_columnconfigure(0,weight=1); return f
    def _atitle(self,p,txt,row,color=ACC):
        ctk.CTkLabel(p,text=txt,font=ctk.CTkFont("Segoe UI",13,"bold"),text_color=color
            ).grid(row=row,column=0,padx=14,pady=(12,4),sticky="w")
    def _abody(self,p,txt,row):
        ctk.CTkLabel(p,text=txt,font=ctk.CTkFont("Segoe UI",11),text_color=DIM,
            justify="left",wraplength=720).grid(row=row,column=0,padx=14,pady=(0,10),sticky="w")
    def _acode(self,p,txt,row,color=GREEN):
        ctk.CTkLabel(p,text=txt,font=ctk.CTkFont("Consolas",12,"bold"),text_color=color
            ).grid(row=row,column=0,padx=20,pady=(0,8),sticky="w")

    def _build_app_t1(self):
        af=self._app_scroll
        ban=ctk.CTkFrame(af,fg_color="#0d1f33",corner_radius=12,border_width=1,border_color=CYAN)
        ban.grid(row=0,column=0,padx=8,pady=(8,6),sticky="ew")
        ban.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(ban,text="Enfriamiento de procesadores — Trayectorias del refrigerante",
            font=ctk.CTkFont("Segoe UI",15,"bold"),text_color=CYAN
            ).grid(row=0,column=0,padx=16,pady=(14,2),sticky="w")
        ctk.CTkLabel(ban,text="Aplicación de ED homogéneas de primer orden en ingeniería de sistemas",
            font=ctk.CTkFont("Segoe UI",10),text_color=DIM
            ).grid(row=1,column=0,padx=16,pady=(0,14),sticky="w")
        fig_a=Figure(figsize=(7,3.0),dpi=96); fig_a.patch.set_facecolor(CARD)
        ax=fig_a.add_subplot(111); self._draw_cooler(ax)
        cva=FigureCanvasTkAgg(fig_a,master=af)
        cva.get_tk_widget().configure(bg=CARD)
        cva.get_tk_widget().grid(row=1,column=0,padx=8,pady=4,sticky="ew"); cva.draw()
        c1=self._acard(af,2)
        self._atitle(c1,"El problema",0,CYAN)
        self._abody(c1,"Cuando un procesador trabaja a plena carga, puede generar más de 100 W/cm². "
            "Si no se disipa ese calor a tiempo, el chip se daña o se apaga por seguridad térmica. "
            "Los sistemas de enfriamiento líquido hacen pasar agua (o un refrigerante glicolado) "
            "por microcanales tallados directamente sobre el disipador de cobre del CPU.",1)
        self._abody(c1,"Para que el diseño sea eficiente, el ingeniero necesita saber por dónde va a "
            "circular cada partícula de líquido. Esas rutas se llaman líneas de corriente "
            "(streamlines), y resultan ser soluciones de una ecuación diferencial.",2)
        c2=self._acard(af,3)
        self._atitle(c2,"¿Cómo aparece la ecuación diferencial?",0)
        self._abody(c2,"La velocidad del fluido en el punto (x, y) del disipador tiene dos componentes: "
            "una horizontal proporcional a M(x,y) y una vertical proporcional a N(x,y). "
            "Para que la curva y(x) sea tangente al vector velocidad en cada punto, se debe cumplir:",1)
        self._acode(c2,"  dy/dx  =  N(x,y) / M(x,y)",2,GREEN)
        self._abody(c2,"Reescribiendo, se obtiene la forma estándar que trabaja este programa:",3)
        self._acode(c2,"  M(x, y) dx  +  N(x, y) dy  =  0",4,CYAN)
        self._abody(c2,"El análisis de dinámica de fluidos computacional (CFD) para un disipador real arrojó "
            "que las componentes de velocidad son proporcionales a:",5)
        self._acode(c2,"  M(x, y) = x² + y²        N(x, y) = −2xy",6,YELLOW)
        self._abody(c2,"Ambas expresiones son homogéneas de grado 2 (si escalas x e y por t, cada función "
            "escala por t²). Eso garantiza que el método de sustitución y = vx va a funcionar "
            "y que la ecuación tiene solución analítica cerrada.",7)
        c3=self._acard(af,4)
        self._atitle(c3,"Resultado",0,GREEN)
        self._abody(c3,"Al resolver la ecuación se obtiene la familia de curvas  x² = C·y, que son parábolas "
            "con vértice en el origen. Cada valor de C da una trayectoria diferente del refrigerante. "
            "El diseñador usa estas curvas para orientar los microcanales y maximizar el contacto "
            "del líquido frío con las zonas más calientes del chip.",1)
        self._acode(c3,"  ( x² + y² ) dx  −  2xy dy  =  0\n  Solución:  x² = C·y",2,CYAN)
        ctk.CTkButton(c3,text="  Resolver este caso",height=40,corner_radius=8,
            fg_color=ACC,hover_color="#2879f0",font=ctk.CTkFont("Segoe UI",12),
            command=self._load_app1
            ).grid(row=3,column=0,padx=14,pady=(4,14),sticky="w")

    def _load_app1(self):
        self._load_t1(
            [("x**2","x²","var"),("+","+","op"),("y**2","y²","var")],
            [("-","−","_neg"),("2","2","d"),("x","x","var"),("y","y","var")])
        self._tabs.set("  Solución Paso a Paso  ")

    def _draw_cooler(self,ax):
        ax.set_xlim(0,12); ax.set_ylim(0,5.5); ax.set_facecolor("#0d1f33"); ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0.5,0.3),11,4.8,
            boxstyle="round,pad=0.15",fc="#101a2e",ec="#30363d",lw=2))
        for xi in np.linspace(0.9,10.6,10):
            ax.add_patch(mpatches.FancyBboxPatch((xi,0.5),0.55,4.2,
                boxstyle="square",fc="#0a2a4a",ec="#1a3a6a",lw=1,alpha=0.8))
        for C_v in [0.5,1.0,2.0,4.0,8.0,16.0]:
            xv=np.linspace(-4.5,4.5,600); yv=xv**2/C_v
            mask=(yv>=0.4)&(yv<=4.9)
            xp=xv[mask]*1.2+6.0; yp=yv[mask]*0.93+0.35
            ok=(xp>=0.6)&(xp<=11.4)
            if ok.sum()>3: ax.plot(xp[ok],yp[ok],color="#79c0ff",alpha=0.6,lw=1.6,zorder=4)
        for xf,yf,xd,yd in [(7.3,3.2,8.0,2.7),(5.0,2.1,4.4,2.5),(6.2,1.2,6.7,1.6)]:
            ax.annotate("",xy=(xd,yd),xytext=(xf,yf),
                arrowprops=dict(arrowstyle="->",color=CYAN,lw=1.4))
        ax.text(6,5.3,"Trayectorias del refrigerante: x² = C·y",
            ha="center",color=CYAN,fontsize=9,fontweight="bold")
        ax.text(6,0.02,"ED:  ( x² + y² ) dx  −  2xy · dy  =  0",
            ha="center",color=YELLOW,fontsize=9,fontweight="bold")
        ax.set_title("Microcanales del disipador de CPU",color=TEXT,fontsize=10,pad=6)

    def _build_app_t2(self):
        af=self._app_scroll
        ban=ctk.CTkFrame(af,fg_color="#150e2e",corner_radius=12,border_width=1,border_color=PURPLE)
        ban.grid(row=0,column=0,padx=8,pady=(8,6),sticky="ew")
        ban.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(ban,text="Circuito RLC — Respuesta transitoria de una señal digital",
            font=ctk.CTkFont("Segoe UI",15,"bold"),text_color=PURPLE
            ).grid(row=0,column=0,padx=16,pady=(14,2),sticky="w")
        ctk.CTkLabel(ban,text="Aplicación de ED homogénea lineal de segundo orden en análisis de circuitos",
            font=ctk.CTkFont("Segoe UI",10),text_color=DIM
            ).grid(row=1,column=0,padx=16,pady=(0,14),sticky="w")
        fig_b=Figure(figsize=(7,3.0),dpi=96); fig_b.patch.set_facecolor(CARD)
        ax=fig_b.add_subplot(111); self._draw_rlc(ax)
        cvb=FigureCanvasTkAgg(fig_b,master=af)
        cvb.get_tk_widget().configure(bg=CARD)
        cvb.get_tk_widget().grid(row=1,column=0,padx=8,pady=4,sticky="ew"); cvb.draw()
        c1=self._acard(af,2)
        self._atitle(c1,"El problema",0,PURPLE)
        self._abody(c1,"Cada vez que tu computadora transmite datos por Ethernet, USB o HDMI, la señal "
            "eléctrica pasa por filtros internos construidos con resistencias, inductores y "
            "capacitores, los llamados circuitos RLC. Su función es dejar pasar las frecuencias "
            "útiles y bloquear el ruido.",1)
        self._abody(c1,"Cuando se desconecta la fuente de alimentación (respuesta libre o transitoria), "
            "la carga almacenada en el capacitor no desaparece de golpe: oscila, se amortigua "
            "o decae dependiendo de los valores de R, L y C. Predecir ese comportamiento es "
            "esencial para diseñar filtros que no distorsionen la señal.",2)
        c2=self._acard(af,3)
        self._atitle(c2,"¿Cómo aparece la ecuación diferencial?",0)
        self._abody(c2,"Aplicando la Ley de Voltajes de Kirchhoff al lazo del circuito sin fuente, la suma "
            "de las caídas de voltaje en R, L y C debe ser cero:",1)
        self._acode(c2,"  L · q″(t)  +  R · q′(t)  +  (1/C) · q(t)  =  0",2,GREEN)
        self._abody(c2,"Aquí q(t) es la carga en el capacitor, q′(t) = i(t) es la corriente, y q″(t) es "
            "la derivada de la corriente. Esta ecuación tiene exactamente la forma:",3)
        self._acode(c2,"  a·y″  +  b·y′  +  c·y  =  0",4,CYAN)
        self._abody(c2,"con  a = L (inductancia),  b = R (resistencia),  c = 1/C. "
            "Es homogénea porque el lado derecho es cero: no hay fuente externa activada.",5)
        c3=self._acard(af,4)
        self._atitle(c3,"Los tres comportamientos posibles",0)
        self._abody(c3,"Todo depende del discriminante  Δ = R² − 4L/C:",1)
        ctk.CTkLabel(c3,
            text="  Δ > 0  →  Sobre-amortiguado:   la carga decae sin oscilar (R muy grande)\n"
                 "  Δ = 0  →  Amortiguamiento crítico:  regresa al equilibrio lo más rápido posible\n"
                 "  Δ < 0  →  Sub-amortiguado:   oscila con amplitud que va decreciendo",
            font=ctk.CTkFont("Consolas",12),text_color=YELLOW,justify="left"
            ).grid(row=2,column=0,padx=20,pady=(0,10),sticky="w")
        c4=self._acard(af,5)
        self._atitle(c4,"Ejemplo concreto: L = 1 H,  R = 2 Ω,  C = 0.25 F",0,GREEN)
        self._abody(c4,"Con esos valores  1/C = 4, así que la ecuación queda:",1)
        self._acode(c4,"  q″ + 2q′ + 4q = 0",2,CYAN)
        self._abody(c4,"El discriminante es  Δ = 4 − 16 = −12 < 0,  circuito sub-amortiguado. "
            "La carga oscila con frecuencia √3 rad/s y su amplitud se reduce como e^(−t). "
            "Este es el comportamiento típico de un filtro resonante de banda estrecha.",3)
        self._acode(c4,"  q(t)  =  e^(−t) · [ C₁·cos(√3·t) + C₂·sin(√3·t) ]",4,YELLOW)
        ctk.CTkButton(c4,text="  Resolver este caso",height=40,corner_radius=8,
            fg_color=ACC,hover_color="#2879f0",font=ctk.CTkFont("Segoe UI",12),
            command=lambda:self._load_t2(1,2,4)
            ).grid(row=5,column=0,padx=14,pady=(4,14),sticky="w")

    def _draw_rlc(self,ax):
        ax.set_xlim(0,10); ax.set_ylim(-1.2,5.0); ax.set_facecolor(CARD); ax.axis("off")
        yt,yb=3.6,0.9; x0,xR1,xR2,xL1,xL2,xC1,xC2,xn=0.5,1.5,3.5,4.2,6.2,6.9,8.9,9.5
        for seg in [([x0,xR1],[yt,yt]),([xR2,xL1],[yt,yt]),([xL2,xC1],[yt,yt]),
                    ([xC2,xn],[yt,yt]),([xn,xn],[yt,yb]),([x0,x0],[yt,yb]),([x0,xn],[yb,yb])]:
            ax.plot(*seg,color=TEXT,lw=2.2)
        nz=8; rxv=np.linspace(xR1,xR2,nz*2+2)
        ryv=np.array([yt+(0.22 if i%2==1 else -0.22) for i in range(len(rxv))]); ryv[0]=ryv[-1]=yt
        ax.plot(rxv,ryv,color=YELLOW,lw=2.0)
        ax.text((xR1+xR2)/2,yt+0.45,"R",ha="center",color=YELLOW,fontsize=14,fontweight="bold")
        ax.text((xR1+xR2)/2,yt-0.55,"Resistencia",ha="center",color=DIM,fontsize=8)
        nc=5; lxv,lyv=[],[]
        for i in range(nc):
            theta=np.linspace(0,np.pi,30); r_i=(xL2-xL1)/(2*nc); cx=xL1+(2*i+1)*r_i
            lxv.extend(cx+r_i*np.cos(theta[::-1])); lyv.extend(yt+r_i*np.sin(theta))
        ax.plot(lxv,lyv,color=CYAN,lw=2.0)
        ax.text((xL1+xL2)/2,yt+0.45,"L",ha="center",color=CYAN,fontsize=14,fontweight="bold")
        ax.text((xL1+xL2)/2,yt-0.55,"Inductor",ha="center",color=DIM,fontsize=8)
        cx=(xC1+xC2)/2; cg=0.22
        ax.plot([xC1,cx-0.01],[yt,yt],color=TEXT,lw=2.2)
        ax.plot([cx-0.01,cx-0.01],[yt-0.40,yt+0.40],color=GREEN,lw=5)
        ax.plot([cx+cg,cx+cg],[yt-0.40,yt+0.40],color=GREEN,lw=5)
        ax.plot([cx+cg,xC2],[yt,yt],color=TEXT,lw=2.2)
        ax.text((xC1+xC2)/2,yt+0.45,"C",ha="center",color=GREEN,fontsize=14,fontweight="bold")
        ax.text((xC1+xC2)/2,yt-0.55,"Capacitor",ha="center",color=DIM,fontsize=8)
        ax.annotate("",xy=(5.5,yt),xytext=(4.5,yt),arrowprops=dict(arrowstyle="->",color=RED,lw=1.8))
        ax.text(5.0,yt+0.62,"i(t) = q'(t)",ha="center",color=RED,fontsize=8.5,fontstyle="italic")
        ax.text(5.0,yb-0.45,"L·q\" + R·q' + (1/C)·q = 0",ha="center",color=TEXT,fontsize=10,fontweight="bold")
        ax.text(5.0,yb-0.80,"equivale a:   a·y\" + b·y' + c·y = 0",ha="center",color=DIM,fontsize=8.5,fontstyle="italic")
        ax.set_title("Circuito RLC en serie — filtro de señal digital",color=TEXT,fontsize=10,pad=6)

    # ─── Solver ──────────────────────────────────────────────────────────────
    def _solve(self):
        if self._solving: return
        self._solving = True
        self._btn_solve.configure(state="disabled", text="  Calculando…")
        self._status.configure(text="⏳  Resolviendo, por favor espera…", text_color=YELLOW)
        self.update()
        def _run():
            try:
                if self._eq_type == "tipo1": self._solve_t1()
                else:                         self._solve_t2()
            except Exception as ex:
                self.after(0, lambda msg=str(ex): self._write_error(f"Error inesperado: {msg}"))
            finally:
                self._solving = False
                self.after(0, lambda: self._btn_solve.configure(
                    state="normal", text="▶  Resolver paso a paso"))
        threading.Thread(target=_run, daemon=True).start()

    # ─── Solver Tipo 1 (mejorado) ─────────────────────────────────────────────
    def _solve_t1(self):
        me = self._fm.get_expr().strip()
        ne = self._fn.get_expr().strip()
        md = self._fm.get_disp().strip()
        nd = self._fn.get_disp().strip()

        if not me or not ne:
            self._write_error(
                "Faltan datos.\n\n"
                "Haz clic en la tarjeta M o N y usa el teclado para ingresar las expresiones."); return

        # ── Parsear ──────────────────────────────────────────────────────────
        try:
            M = sp.sympify(me, locals={"x": xs, "y": ys})
            N = sp.sympify(ne, locals={"x": xs, "y": ys})
        except Exception:
            self._write_error(
                f"No se pudo interpretar la expresión.\n"
                f"  M ingresada: {repr(me)}\n"
                f"  N ingresada: {repr(ne)}\n\n"
                f"Revisa que no haya operadores duplicados o paréntesis sin cerrar."); return

        # ── Verificar que solo usa x e y (sin t, Laplace, etc.) ──────────────
        syms_M = M.free_symbols; syms_N = N.free_symbols
        allowed = {xs, ys}
        extra = (syms_M | syms_N) - allowed
        if extra:
            nombres = ", ".join(str(s) for s in sorted(extra, key=str))
            self._write_error(
                f"La expresión contiene variables no permitidas: {nombres}\n\n"
                f"Este programa solo acepta ecuaciones diferenciales homogéneas\n"
                f"de 1er orden en las variables x e y.\n\n"
                f"No se aceptan: transformadas de Laplace, variables t, s, z, etc."); return

        HL = "─" * 64
        L  = []

        L += ["── ED HOMOGÉNEA DE PRIMER ORDEN ─────────────────────────────", "",
              "  Ecuación ingresada:",
              f"    ( {md} ) dx  +  ( {nd} ) dy  =  0", ""]

        # ── PASO 1: verificar homogeneidad ───────────────────────────────────
        L += [HL, "  PASO 1  Verificar que la ecuación es homogénea", HL, ""]
        L += ["  Una ecuación M dx + N dy = 0 es homogénea de grado n si",
              "  al sustituir x → t·x  e  y → t·y,  tanto M como N escalan",
              "  por el mismo factor t^n:", "",
              "    M(t·x, t·y) = t^n · M(x, y)",
              "    N(t·x, t·y) = t^n · N(x, y)", "",
              "  Realizamos la sustitución:", ""]

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
            L += ["  Al sustituir  x → t·x,  y → t·y  en M y N, no es posible",
                  "  factorizar un mismo t^n en ambas funciones.", "",
                  "  ✗  La ecuación ingresada NO es homogénea.", "",
                  "  ─────────────────────────────────────────────────────────",
                  "   ESTE PROGRAMA SOLO RESUELVE ECUACIONES HOMOGÉNEAS",
                  "  ─────────────────────────────────────────────────────────", "",
                  "  Revisa tu ecuación e intenta de nuevo."]
            self._write_out(L)
            self.after(0, lambda: self._status.configure(
                text="✗  Ecuación no homogénea.", text_color=RED))
            return

        Mh = _sp2h(sp.factor(sp.simplify(Mt / ts**g)))
        Nh = _sp2h(sp.factor(sp.simplify(Nt / ts**g)))
        L += [f"  Para M:  M(t·x, t·y)  =  t^{g} · ( {Mh} )  =  t^{g} · M(x,y)  ✓",
              f"  Para N:  N(t·x, t·y)  =  t^{g} · ( {Nh} )  =  t^{g} · N(x,y)  ✓", "",
              f"  Las dos funciones son homogéneas de grado n = {g}.",
              "  La ecuación es homogénea y puede resolverse con y = v·x.", ""]

        # ── PASO 2 ───────────────────────────────────────────────────────────
        L += [HL, "  PASO 2  Sustitución  y = v·x", HL, ""]
        L += ["  Introducimos el cambio de variable:", "",
              "    y  =  v·x",
              "    dy =  v dx + x dv", ""]

        # ── PASO 3 ───────────────────────────────────────────────────────────
        L += [HL, "  PASO 3  Sustituir en la ecuación", HL, ""]
        Mv = sp.expand(M.subs(ys, vs*xs))
        Nv = sp.expand(N.subs(ys, vs*xs))
        # Extraer potencia de x para mostrar más limpio
        Mv_f = sp.factor(Mv)
        Nv_f = sp.factor(Nv)
        coef_dx = sp.factor(sp.expand(Mv + vs*Nv))
        L += [f"    M(x, v·x)  =  {_sp2h(Mv_f)}",
              f"    N(x, v·x)  =  {_sp2h(Nv_f)}", "",
              f"    [{_sp2h(coef_dx)}] dx  +  [{_sp2h(Nv_f)}·x] dv  =  0", ""]

        # ── PASO 4: separar variables ─────────────────────────────────────────
        L += [HL, "  PASO 4  Separar variables", HL, ""]
        # Evaluamos en x=1 para obtener la función solo en v
        M1 = sp.simplify(M.subs([(xs, 1), (ys, vs)]))
        N1 = sp.simplify(N.subs([(xs, 1), (ys, vs)]))
        A  = sp.simplify(M1 + vs*N1)   # coeficiente de dx/x
        B  = sp.simplify(N1)            # coeficiente de dv

        if A == 0:
            L += ["  El coeficiente de dx/x resultó ser cero.",
                  "  La ecuación tiene una forma especial; revísala manualmente."]
            self._write_out(L); return

        # Integrando: -B/A respecto de v
        ig_raw = sp.cancel(-B / A)
        ig_pf  = sp.apart(ig_raw, vs)   # fracciones parciales

        L += [f"    dx/x  =  ({_sp2h(ig_pf)}) dv", ""]

        # ── PASO 5: integrar ──────────────────────────────────────────────────
        L += [HL, "  PASO 5  Integrar ambos lados", HL, ""]

        # Intentamos integrar con timeout generoso y cascada de métodos
        itg_result, ok = _integrate_safe(ig_pf, vs, timeout_sec=14)
        # Si falló con apart, intentar con la expresión original cancelada
        if not ok:
            itg_result, ok = _integrate_safe(ig_raw, vs, timeout_sec=10)

        if not ok:
            # Mostrar mensaje informativo limpio en la UI en lugar de colgar
            L += [f"  La integral  ∫ ( {_sp2h(ig_pf)} ) dv  no tiene forma",
                  "  cerrada que SymPy pueda calcular dentro del tiempo límite.", "",
                  "  Esto ocurre porque el integrando no es racionalmente integrable",
                  "  o requiere funciones especiales (hipergeométricas, elípticas, etc.).", "",
                  "  Resultado hasta aquí:",
                  f"    ln|x|  =  ∫ ( {_sp2h(ig_pf)} ) dv  +  C   [donde v = y/x]", "",
                  "  Puedes evaluar esa integral con una tabla, WolframAlpha o Mathematica."]
            self._write_out(L)
            self._plot_t1(M, N)
            return

        itg = itg_result
        L += [f"    ln|x|  =  {_sp2h(itg)}  +  C₁", ""]

        # ── PASO 6: retrosustitución y despeje explícito ──────────────────────
        L += [HL, "  PASO 6  Retrosustitución  v = y/x  y despeje de y", HL, ""]
        sol_raw = itg.subs(vs, ys/xs)
        sol_raw = sp.simplify(sol_raw)

        Cs = sp.Symbol("C")   # constante genérica para despejar

        arg_expr = None        # el argumento del log combinado
        implícita_str = ""

        try:
            lhs   = sp.logcombine(sp.log(xs) - sol_raw, force=True)
            lhs_s = sp.simplify(lhs)
            if lhs_s.func == sp.log:
                arg_expr = lhs_s.args[0]
            else:
                raise ValueError
        except Exception:
            pass

        if arg_expr is not None:
            arg_h = _sp2h(arg_expr)
            L += [f"    Sustituyendo v = y/x y combinando logaritmos:", "",
                  f"    ln( {arg_h} )  =  C", ""]

            # ── Limpiar fracciones multiplicando por denominador (si no tiene y) ──
            arg_clean = sp.together(arg_expr)
            num, den  = sp.fraction(arg_clean)
            if den != 1 and not den.has(ys):
                # ambos lados × den: num = C · den (C absorbe la constante)
                lhs_impl = sp.expand(num)
                eq_impl  = sp.Eq(lhs_impl, Cs * sp.expand(den))
                implícita_str = f"{_sp2h(lhs_impl)}  =  C·{_sp2h(sp.expand(den))}"
                L += [f"  Multiplicando ambos lados por ({_sp2h(sp.expand(den))})",
                      f"  (C absorbe el factor constante):","",
                      f"    {implícita_str}", ""]
            else:
                eq_impl = sp.Eq(arg_expr, Cs)
                implícita_str = f"{arg_h}  =  C"

            # ── Intentar despejar y explícitamente ───────────────────────────
            despejado = False
            try:
                sols_y = sp.solve(eq_impl, ys)
                if sols_y:
                    sols_y = [sp.simplify(s) for s in sols_y]
                    # Filtrar soluciones reales (descartar complejas puras)
                    sols_y = [s for s in sols_y if not s.has(sp.I)] or sols_y
                    L += ["  Despejando y:",""]
                    ancho = max(len(_sp2h(s)) for s in sols_y) + 6
                    borde = "═" * (ancho + 6)
                    L += [f"  ╔{borde}╗"]
                    for s in sols_y:
                        sh = _sp2h(s)
                        L += [f"  ║   y  =  {sh:<{ancho}}  ║"]
                    L += [f"  ╚{borde}╝", ""]
                    if len(sols_y) > 1:
                        L += ["  (Ambas ramas son soluciones válidas según la condición inicial.)", ""]
                    despejado = True
            except Exception:
                pass

            # ── Si no se pudo despejar y, intentar despejar x ─────────────────
            if not despejado:
                try:
                    sols_x = sp.solve(eq_impl, xs)
                    if sols_x:
                        sols_x = [sp.simplify(s) for s in sols_x]
                        sols_x = [s for s in sols_x if not s.has(sp.I)] or sols_x
                        L += ["  No es posible despejar y explícitamente.",
                              "  Despejando x:", ""]
                        ancho = max(len(_sp2h(s)) for s in sols_x) + 6
                        borde = "═" * (ancho + 6)
                        L += [f"  ╔{borde}╗"]
                        for s in sols_x:
                            sh = _sp2h(s)
                            L += [f"  ║   x  =  {sh:<{ancho}}  ║"]
                        L += [f"  ╚{borde}╝", ""]
                        despejado = True
                except Exception:
                    pass

            # ── Fallback: forma implícita ─────────────────────────────────────
            if not despejado:
                ancho = len(implícita_str) + 4
                borde = "═" * (ancho + 4)
                L += ["  Solución implícita:", "",
                      f"  ╔{borde}╗",
                      f"  ║   {implícita_str}   ║",
                      f"  ╚{borde}╝", ""]

        else:
            # logcombine no logró combinar: mostrar forma directa
            sol_h = _sp2h(sol_raw)
            L += [f"    ln|x|  =  {sol_h}  +  C", "",
                  f"  ╔══════════════════════════════════════════════════╗",
                  f"  ║   ln|x| − ( {sol_h} )  =  C              ║",
                  f"  ╚══════════════════════════════════════════════════╝", ""]

        L += ["  C es la constante arbitraria de integración.", "", "─" * 64]
        self._write_out(L)
        self._plot_t1(M, N)

    # ─── Solver Tipo 2 (sin cambios) ─────────────────────────────────────────
    def _solve_t2(self):
        try:
            af = float(self._t2f["a"].get_expr() or "0")
            bf = float(self._t2f["b"].get_expr() or "0")
            cf = float(self._t2f["c"].get_expr() or "0")
        except ValueError:
            self._write_error("Los coeficientes solo pueden ser números.\nEjemplos: 1  |  −2  |  0.5"); return
        if af == 0:
            self._write_error("El coeficiente  a  no puede ser cero."); return
        a = sp.Rational(af).limit_denominator(1000)
        b = sp.Rational(bf).limit_denominator(1000)
        c = sp.Rational(cf).limit_denominator(1000)
        HL = "─" * 64; L = []
        L += ["── ED HOMOGÉNEA LINEAL DE SEGUNDO ORDEN ──────────────────────", "",
              "  Ecuación ingresada:", f"    {self._fmt_eq(af,bf,cf)}  =  0", ""]
        L += [HL, "  PASO 1  Identificar los coeficientes", HL, ""]
        L += [f"    a  =  {a}    coeficiente de y″",
              f"    b  =  {b}    coeficiente de y′",
              f"    c  =  {c}    coeficiente de y", ""]
        L += [HL, "  PASO 2  Plantear la ecuación característica", HL, ""]
        L += ["  Proponemos  y = e^(r·t):",
              "    y′  =  r · e^(r·t)", "    y″  =  r² · e^(r·t)", "",
              f"    {self._fmt_char(af,bf,cf)}  =  0", ""]
        disc = b**2 - 4*a*c; df = float(disc)
        L += [HL, "  PASO 3  Discriminante  Δ = b² − 4ac", HL, ""]
        L += [f"    Δ  =  ({b})²  −  4·({a})·({c})  =  {disc}", ""]
        r1s = sp.simplify((-b + sp.sqrt(disc)) / (2*a))
        r2s = sp.simplify((-b - sp.sqrt(disc)) / (2*a))
        L += [HL, "  PASO 4  Raíces", HL, ""]
        L += [f"    r₁  =  {_sp2h(r1s)}", f"    r₂  =  {_sp2h(r2s)}", ""]
        L += [HL, "  PASO 5  Solución general", HL, ""]
        if df > 0:
            cs = "dist"; r1n = float(r1s); r2n = float(r2s); alp = bet = None
            L += [f"  Δ > 0  →  Raíces reales distintas", "",
                  f"  ┌────────────────────────────────────────────────────────────┐",
                  f"  │  y(t)  =  C₁·e^({_sp2h(r1s)}·t)  +  C₂·e^({_sp2h(r2s)}·t)         │",
                  f"  └────────────────────────────────────────────────────────────┘", ""]
        elif df == 0:
            cs = "rep"; r1n = r2n = float(r1s); alp = bet = None
            L += [f"  Δ = 0  →  Raíz real repetida  r = {_sp2h(r1s)}", "",
                  f"  ┌────────────────────────────────────────────────────────────┐",
                  f"  │  y(t)  =  ( C₁  +  C₂·t ) · e^({_sp2h(r1s)}·t)               │",
                  f"  └────────────────────────────────────────────────────────────┘", ""]
        else:
            cs = "comp"
            als = sp.simplify(-b/(2*a)); bes = sp.simplify(sp.sqrt(-disc)/(2*a))
            alp = float(als); bet = float(bes)
            r1n = complex(alp, bet); r2n = complex(alp, -bet)
            L += [f"  Δ < 0  →  Raíces complejas   α = {_sp2h(als)},  β = {_sp2h(bes)}", "",
                  f"  ┌────────────────────────────────────────────────────────────┐",
                  f"  │  y(t) = e^({_sp2h(als)}·t)·[ C₁·cos({_sp2h(bes)}·t) + C₂·sin({_sp2h(bes)}·t) ] │",
                  f"  └────────────────────────────────────────────────────────────┘", ""]
        L += ["  C₁ y C₂ se determinan con las condiciones iniciales  y(0) y y′(0).", "", "─" * 64]
        self._write_out(L)
        self._plot_t2(cs, r1n, r2n, alp, bet)

    # ─── Gráficas ─────────────────────────────────────────────────────────────
    def _sax(self, ax, title, xl="x", yl="y"):
        ax.set_facecolor("#0d1117")
        ax.set_title(title, color=TEXT, fontsize=11, pad=10, fontweight="bold")
        ax.set_xlabel(xl, color=DIM, fontsize=10)
        ax.set_ylabel(yl, color=DIM, fontsize=10)
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
        xv = np.linspace(-4, 4, 20); yv = np.linspace(-4, 4, 20)
        Xg, Yg = np.meshgrid(xv, yv)
        plot_data = None
        try:
            Mf = sp.lambdify((xs, ys), M, "numpy")
            Nf = sp.lambdify((xs, ys), N, "numpy")
            with np.errstate(all="ignore"):
                Mv = np.array(Mf(Xg, Yg), dtype=float)
                Nv = np.array(Nf(Xg, Yg), dtype=float)
                U = np.where(np.isfinite(Nv), Nv, 0.)
                V = np.where(np.isfinite(Mv), -Mv, 0.)
                nm = np.sqrt(U**2 + V**2); nm[nm == 0] = 1
                U /= nm; V /= nm
                mask = np.isfinite(U) & np.isfinite(V)
            plot_data = (Xg, Yg, U, V, mask)
        except Exception as e:
            plot_data = e

        def _draw():
            self._fig.clear(); self._fig.patch.set_facecolor(INP)
            ax = self._fig.add_subplot(111)
            self._sax(ax, "Campo de direcciones  —  dy/dx = −M/N")
            if isinstance(plot_data, Exception):
                ax.text(0.5, 0.5, f"No se pudo graficar:\n{plot_data}",
                    ha="center", va="center", color=RED, transform=ax.transAxes)
            elif plot_data:
                Xg, Yg, U, V, mask = plot_data
                ax.quiver(Xg[mask], Yg[mask], U[mask], V[mask],
                    color=CYAN, alpha=0.72, scale=26, width=0.003,
                    headwidth=4, headlength=5)
                ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
            self._fig.tight_layout(pad=2.0); self._cv.draw()
        self.after(0, _draw)

    def _plot_t2(self, cs, r1, r2, alp, bet):
        xp = np.linspace(0, 8, 600)
        ics = [(1, 0), (0, 1), (1, 0.5), (1, -0.5), (0.5, 0.5)]
        curves = []
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
                    yp  = np.exp(alp*xp) * (c1*np.cos(bet*xp) + c2*np.sin(bet*xp))
            mask = np.isfinite(yp) & (np.abs(yp) < 15)
            if mask.any(): curves.append((xp[mask], yp[mask], PAL[i], c1, c2))
        titles = {"dist": "Caso 1 — Sobre-amortiguado",
                  "rep":  "Caso 2 — Amortiguamiento crítico",
                  "comp": "Caso 3 — Sub-amortiguado (oscilaciones)"}
        def _draw():
            self._fig.clear(); self._fig.patch.set_facecolor(INP)
            ax = self._fig.add_subplot(111)
            self._sax(ax, f"Respuesta transitoria  ·  {titles[cs]}",
                xl="t  (tiempo)", yl="y(t)  (señal / carga)")
            for (xd, yd, col, c1, c2) in curves:
                ax.plot(xd, yd, color=col, lw=1.8, alpha=0.9,
                    label=f"y(0)={c1},  y'(0)={c2}")
            ax.set_xlim(0, 8); ax.set_ylim(-6, 6)
            ax.axhline(0, color=BDR, lw=1.0, ls="--", alpha=0.6)
            ax.legend(fontsize=9, facecolor="#161b22", labelcolor=TEXT,
                edgecolor=BDR, loc="best", framealpha=0.92)
            self._fig.tight_layout(pad=2.0); self._cv.draw()
        self.after(0, _draw)

    def _fmt_eq(self, a, b, c):
        parts = []
        for val, term in [(a,"y″"), (b,"y′"), (c,"y")]:
            if val == 0: continue
            if not parts:
                if val == 1:   parts.append(term)
                elif val ==-1: parts.append(f"−{term}")
                else:           parts.append(f"{val:g}{term}")
            else:
                if val == 1:   parts.append(f"+ {term}")
                elif val ==-1: parts.append(f"− {term}")
                elif val > 0:  parts.append(f"+ {val:g}{term}")
                else:           parts.append(f"− {abs(val):g}{term}")
        return " ".join(parts) or "0"

    def _fmt_char(self, a, b, c):
        parts = []
        for val, term in [(a,"r²"), (b,"r"), (c,"")]:
            if val == 0: continue
            d = f"{val:g}{term}" if term else f"{val:g}"
            if not parts:   parts.append(d)
            elif val > 0:   parts.append(f"+ {d}")
            else:            parts.append(f"− {abs(val):g}{term}")
        return " ".join(parts)

    # ─── Salida de texto ──────────────────────────────────────────────────────
    def _write_out(self, lines):
        def _do():
            self._out.configure(state="normal")
            self._out.delete("1.0", "end")
            for line in lines: self._out.insert("end", line + "\n")
            self._out.see("1.0")
            self._out.configure(state="disabled")
            self._tabs.set("  Solución Paso a Paso  ")
            self._status.configure(
                text="Listo. Ve a 'Gráfica de Solución' para ver la gráfica.",
                text_color=GREEN)
        self.after(0, _do)

    def _write_error(self, msg):
        def _do():
            self._out.configure(state="normal")
            self._out.delete("1.0", "end")
            self._out.insert("end", f"\n  ✗  {msg}\n")
            self._out.configure(state="disabled")
            self._tabs.set("  Solución Paso a Paso  ")
            self._status.configure(text="Revisa los datos ingresados.", text_color=RED)
        self.after(0, _do)

    # ─── Limpiar todo (fix completo) ──────────────────────────────────────────
    def _clear_all(self):
        if self._solving: return   # no limpiar mientras corre el solver
        self._show_welcome()
        self._blank_graph()
        self._status.configure(text="")
        # Limpiar campos del tipo activo sin romper referencias
        if self._eq_type == "tipo1":
            if hasattr(self, "_fm"): self._fm.clear()
            if hasattr(self, "_fn"): self._fn.clear()
        else:
            if hasattr(self, "_t2f"):
                for k in "abc":
                    if k in self._t2f: self._t2f[k].clear()
        self._preview()

    def _show_welcome(self):
        lines = [
            "  Solucionador de ED Homogéneas",
            "  ─────────────────────────────────────────────────────────────", "",
            "  Cómo usar:", "",
            "    1. Elige el tipo con el selector del panel izquierdo",
            "    2. Haz clic en la tarjeta del campo que quieras llenar",
            "    3. Usa el teclado de pantalla o el físico",
            "    4. Presiona  ▶ Resolver paso a paso", "", "─" * 64]
        self._out.configure(state="normal")
        self._out.delete("1.0", "end")
        for line in lines: self._out.insert("end", line + "\n")
        self._out.see("1.0")
        self._out.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()