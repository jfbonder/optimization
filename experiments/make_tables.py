"""Generate the LaTeX tables of paper 2 from the JSON results."""
import json
import os

from common import RES, ROOT

TAB = os.path.join(ROOT, "paper2_steklov_shape_derivative", "tables")
os.makedirs(TAB, exist_ok=True)

names = {"disk": r"unit disk, $\alpha=\pi/2$",
         "lshape": r"L-shaped domain, $\alpha=1/2$",
         "handle": r"two disks with a handle, $\alpha=\pi$",
         "bigdisk": r"disk $B_3$, $\alpha=\pi/4$"}

# --- domains -----------------------------------------------------------------
rows = []
for key, label in names.items():
    f = os.path.join(RES, f"holes_{key}.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    rows.append(
        f"{label} & {d['nodes']} & {d['lambda_initial']:.5f} & {d['swap']['lam']:.5f} "
        f"& {d['rearrangement']['lam']:.5f} & {d['flow']['lam']:.5f} & {d['flow']['remeshes']} \\\\")
with open(os.path.join(TAB, "domains.tex"), "w") as fh:
    fh.write("\\begin{tabular}{lrrrrrr}\n\\hline\n"
             "domain & nodes & initial & Alg.~\\ref{alg.swap} & Alg.~\\ref{alg.rearr} & Alg.~\\ref{alg.flow} & remeshes\\\\\n\\hline\n")
    fh.write("\n".join(rows) + "\n\\hline\n\\end{tabular}\n")

# --- convergence ---------------------------------------------------------------
f = os.path.join(RES, "convergence_lshape.json")
if os.path.exists(f):
    conv = json.load(open(f))
    old = {0.2: None, 0.1: "0.50227"}
    with open(os.path.join(TAB, "convergence.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrrr}\n\\hline\n"
                 "$h$ & nodes & $|A^*_h|$ & $\\lambda_{1,h}$ (Alg.~\\ref{alg.rearr}) & nodes & $|A_h|$ & $\\lambda_{1,h}$ (Alg.~\\ref{alg.flow})\\\\\n\\hline\n")
        for r in conv:
            fh.write(f"{r['h']:g} & {r['nodes']} & {r['area_rearr']:.4f} & {r['lam_rearr']:.5f} & "
                     f"{r['nodes_flow']} & {r['area_flow']:.4f} & {r['lam_flow']:.5f}\\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")

# --- translation ---------------------------------------------------------------
f = os.path.join(RES, "translation.json")
if os.path.exists(f):
    import sys
    sys.path.insert(0, ROOT)
    from steklov_holes import radial
    tr = json.load(open(f))
    with open(os.path.join(TAB, "translation.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrrr}\n\\hline\n"
                 "$R$ & $r$ & $R^*(r)$ & $\\lambda_1(R,r)$ & $\\lambda_{1,h}(B_r)$ & "
                 "$\\frac{d^2}{dt^2}\\lambda_1(B_r(te_1))|_{t=0}$ & FEM\\\\\n\\hline\n")
        for d in tr:
            exact2 = radial.translation_second_derivative(d["R"], d["r"])
            fh.write(f"{d['R']:g} & {d['r']:g} & {d['R_star']:.4f} & {d['lambda_exact']:.6f} & "
                     f"{d['lambda_h0']:.6f} & ${exact2:+.3e}$ & ${d['lambda_h_second_derivative']:+.3e}$\\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
print("tables written to", TAB)
