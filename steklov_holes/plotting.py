"""Figures and movies of discrete holes and eigenfunctions.

Convention: the hole is always shown in a binary panel ({u = 0} black,
{u > 0} white, boundary of Omega outlined); the eigenfunction is shown in a
separate panel.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection   # noqa: E402
from matplotlib.tri import Triangulation            # noqa: E402

HOLE_COLOR = "#111111"


def _finish(ax, p):
    ax.set_aspect("equal")
    pad = 0.03 * np.ptp(p, axis=0).max()
    ax.set_xlim(p[:, 0].min() - pad, p[:, 0].max() + pad)
    ax.set_ylim(p[:, 1].min() - pad, p[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _boundary_segments(p, t):
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    b = u[cnt == 1]
    return p[b]


def draw_hole(ax, p, t, hole, show_mesh=False):
    """Binary picture: hole (u = 0) in black, Omega \\ A in white."""
    ax.add_collection(PolyCollection(p[t[~hole]], facecolors="white",
                                     edgecolors="0.8" if show_mesh else "white",
                                     lw=0.15 if show_mesh else 0.3))
    ax.add_collection(PolyCollection(p[t[hole]], facecolors=HOLE_COLOR, edgecolors=HOLE_COLOR, lw=0.3))
    ax.add_collection(LineCollection(_boundary_segments(p, t), colors="k", lw=1.0))
    _finish(ax, p)


def draw_eigenfunction(ax, p, t, u, vmax=None, cmap="viridis", contours=True):
    tri = Triangulation(p[:, 0], p[:, 1], t)
    vmax = float(np.max(u)) if vmax is None else vmax
    im = ax.tripcolor(tri, u, shading="gouraud", cmap=cmap, vmin=0, vmax=vmax)
    if contours and vmax > 0:
        ax.tricontour(tri, u, levels=np.linspace(0, vmax, 11)[1:-1], colors="w", linewidths=0.4, alpha=0.7)
    ax.add_collection(LineCollection(_boundary_segments(p, t), colors="k", lw=0.8))
    _finish(ax, p)
    return im


def plot_result(res, fname=None, title=None, show_mesh=False):
    m = res.mesh
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.9))
    draw_hole(axs[0], m.p, m.t, res.hole, show_mesh)
    axs[0].set_title(r"hole $\{u_h=0\}$" + f"   $|A_h|={res.area:.4f}$", fontsize=10)
    im = draw_eigenfunction(axs[1], m.p, m.t, res.u)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02)
    axs[1].set_title(rf"eigenfunction $u_h$,  $\lambda_{{1,h}}={res.lam:.5f}$", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200)
        plt.close(fig)
    return fig


def plot_holes_panel(states, titles, fname, show_mesh=False, size=2.8, with_eigenfunction=True):
    """Columns = states; first row holes (binary), second row eigenfunctions."""
    n = len(states)
    nrows = 2 if with_eigenfunction else 1
    fig, axs = plt.subplots(nrows, n, figsize=(size * n, size * nrows), squeeze=False)
    vmax = max(float(np.max(s[3])) for s in states)
    for j, ((p, t, hole, u), ttl) in enumerate(zip(states, titles)):
        draw_hole(axs[0, j], p, t, hole, show_mesh)
        axs[0, j].set_title(ttl, fontsize=9)
        if with_eigenfunction:
            draw_eigenfunction(axs[1, j], p, t, u, vmax=vmax)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def make_movie(frames, fname, fps=8, title="", show_mesh=False, max_frames=400, dpi=110,
               hold_last=2.0):
    """MP4 (or GIF, by extension) of the iterations: hole | eigenfunction | lambda."""
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if len(frames) > max_frames:
        idx = np.unique(np.linspace(0, len(frames) - 1, max_frames).astype(int))
        frames = [frames[i] for i in idx]
    lams = np.array([f["lam"] for f in frames])
    vmax = max(float(np.max(f["u"])) for f in frames)

    fig = plt.figure(figsize=(12.8, 4.2))
    ax0 = fig.add_axes([0.01, 0.04, 0.30, 0.80])
    ax1 = fig.add_axes([0.33, 0.04, 0.30, 0.80])
    ax2 = fig.add_axes([0.70, 0.14, 0.28, 0.70])
    if fname.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p"])
    lo, hi = lams.min(), lams.max()
    pad = 0.05 * (hi - lo + 1e-12)
    seq = list(range(len(frames))) + [len(frames) - 1] * int(hold_last * fps)
    with writer.saving(fig, fname, dpi=dpi):
        for k in seq:
            f = frames[k]
            for ax in (ax0, ax1, ax2):
                ax.clear()
            draw_hole(ax0, f["p"], f["t"], f["hole"], show_mesh)
            ax0.set_title(rf"hole $\{{u_h=0\}}$   $|A_h|={f['area']:.4f}$", fontsize=9)
            draw_eigenfunction(ax1, f["p"], f["t"], f["u"], vmax=vmax)
            ax1.set_title(r"eigenfunction $u_h$", fontsize=9)
            # the curve is drawn progressively, together with the hole
            ax2.plot(np.arange(k + 1), lams[:k + 1], color="C0", lw=1.5, marker="." if len(lams) < 80 else None)
            ax2.plot([k], [lams[k]], "o", color="C3", ms=5)
            ax2.set_xlim(-0.5, len(lams) - 0.5)
            ax2.set_ylim(lo - pad, hi + pad)
            ax2.set_xlabel("iteration")
            ax2.set_ylabel(r"$\lambda$")
            ax2.set_title(rf"$\lambda={f['lam']:.5f}$", fontsize=9)
            ax2.grid(alpha=0.3)
            fig.suptitle(f"{title}   —   {f['label']}, iteration {k}", fontsize=10)
            writer.grab_frame()
    plt.close(fig)
