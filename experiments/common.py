"""Shared helpers for the experiment scripts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "paper2_steklov_shape_derivative", "figures")
MOV = os.path.join(ROOT, "movies")
RES = os.path.join(ROOT, "experiments", "results")
for d in (FIG, MOV, RES):
    os.makedirs(d, exist_ok=True)
