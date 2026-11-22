"""Regenerate the movies from the saved frames (experiments/results/frames_*.pkl.gz)."""
import glob
import gzip
import os
import pickle

from common import MOV, RES
from steklov_holes import plotting as pl

for f in sorted(glob.glob(f"{RES}/frames_*.pkl.gz")):
    name = os.path.basename(f)[len("frames_"):-len(".pkl.gz")]
    d = pickle.load(gzip.open(f))
    pl.make_movie(d["original"], f"{MOV}/{name}_original_algorithm.mp4", fps=15,
                  title=d["title"] + " — original algorithm")
    pl.make_movie(d["new"], f"{MOV}/{name}_rearrangement_and_flow.mp4", fps=4,
                  title=d["title"] + " — new method")
    print("movies for", name)
