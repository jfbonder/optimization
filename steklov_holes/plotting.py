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


def make_comparison_movie(frames_a, frames_b, fname, labels=("original algorithm", "new method"),
                          title="", fps=10, dpi=100, hold_last=2.0, max_frames=300):
    """Side-by-side movie: row 1 = method A, row 2 = method B (hole | eigenfunction),
    right panel: both eigenvalue curves drawn progressively on the same axes.
    Frame k shows iteration min(k, n-1) of each method."""
    from matplotlib.animation import FFMpegWriter, PillowWriter

    def sub(fr):
        if len(fr) > max_frames:
            idx = np.unique(np.linspace(0, len(fr) - 1, max_frames).astype(int))
            return [fr[i] for i in idx], idx
        return fr, np.arange(len(fr))

    fa, ia = sub(frames_a)
    fb, ib = sub(frames_b)
    la = np.array([f["lam"] for f in fa])
    lb = np.array([f["lam"] for f in fb])
    vmax = max(float(np.max(f["u"])) for f in fa + fb)
    lo, hi = min(la.min(), lb.min()), max(la.max(), lb.max())
    pad = 0.05 * (hi - lo + 1e-12)
    nmax = max(ia[-1], ib[-1]) + 1

    fig = plt.figure(figsize=(13.0, 7.2))
    axh = [fig.add_axes([0.01, 0.52, 0.22, 0.40]), fig.add_axes([0.01, 0.04, 0.22, 0.40])]
    axe = [fig.add_axes([0.24, 0.52, 0.22, 0.40]), fig.add_axes([0.24, 0.04, 0.22, 0.40])]
    ax2 = fig.add_axes([0.55, 0.12, 0.42, 0.74])
    if fname.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    n = max(len(fa), len(fb))
    seq = list(range(n)) + [n - 1] * int(hold_last * fps)
    colors = ("C1", "C0")
    with writer.saving(fig, fname, dpi=dpi):
        for k in seq:
            ka, kb = min(k, len(fa) - 1), min(k, len(fb) - 1)
            for ax in axh + axe + [ax2]:
                ax.clear()
            for row, (fr, kk, idx, lab, col) in enumerate(((fa, ka, ia, labels[0], colors[0]),
                                                          (fb, kb, ib, labels[1], colors[1]))):
                f = fr[kk]
                draw_hole(axh[row], f["p"], f["t"], f["hole"])
                axh[row].set_title(f"{lab}: hole, it. {idx[kk]}", fontsize=9, color=col)
                draw_eigenfunction(axe[row], f["p"], f["t"], f["u"], vmax=vmax)
                axe[row].set_title(rf"$u_h$,  $\lambda={f['lam']:.5f}$", fontsize=9)
            ax2.plot(ia[:ka + 1], la[:ka + 1], color=colors[0], lw=1.5, label=labels[0])
            ax2.plot(ib[:kb + 1], lb[:kb + 1], color=colors[1], lw=1.5, label=labels[1])
            ax2.plot([ia[ka]], [la[ka]], "o", color=colors[0], ms=5)
            ax2.plot([ib[kb]], [lb[kb]], "o", color=colors[1], ms=5)
            ax2.set_xlim(-0.5, nmax - 0.5)
            ax2.set_ylim(lo - pad, hi + pad)
            ax2.set_xlabel("iteration")
            ax2.set_ylabel(r"$\lambda_{1,h}$")
            ax2.grid(alpha=0.3)
            ax2.legend(loc="upper right", fontsize=9)
            fig.suptitle(title, fontsize=11)
            writer.grab_frame()
    plt.close(fig)


def make_pipeline_movie(frames_a, frames_b, fname, labels=("Alg. 1 \u2192 Alg. 3", "Alg. 2 \u2192 Alg. 3"),
                        totals=None, title="", fps=12, duration=20.0, dpi=100, hold_last=3.0):
    """Real-time comparison of two pipelines.

    Every frame carries a timestamp ``time`` (seconds of computation, recording
    overhead excluded).  Computation time is rescaled so that the whole movie
    lasts ``duration`` seconds (normalized time): at movie time s the panels
    show the last state computed before tau = s * T / duration, where T is the
    total time of the slower pipeline.  Panels: hole of pipeline A, hole of pipeline B, and both eigenvalue
    curves against computation time (dotted line: switch to the shape flow).
    ``totals`` = (total time A, total time B); by default the last timestamps.
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    def switch(fr):
        for i, f in enumerate(fr):
            if "flow" in f["label"]:
                return i
        return None

    ta = np.array([f["time"] for f in frames_a])
    tb = np.array([f["time"] for f in frames_b])
    la = np.array([f["lam"] for f in frames_a])
    lb = np.array([f["lam"] for f in frames_b])
    Ta, Tb = totals if totals is not None else (ta[-1], tb[-1])
    T = max(Ta, Tb)
    sa, sb = switch(frames_a), switch(frames_b)
    lo, hi = min(la.min(), lb.min()), max(la.max(), lb.max())
    pad = 0.05 * (hi - lo + 1e-12)

    fig = plt.figure(figsize=(14.5, 5.0))
    ax_a = fig.add_axes([0.01, 0.06, 0.28, 0.76])
    ax_b = fig.add_axes([0.30, 0.06, 0.28, 0.76])
    ax2 = fig.add_axes([0.65, 0.14, 0.33, 0.68])
    if fname.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    n = max(2, int(fps * duration))
    taus = list(np.linspace(0, T, n)) + [T] * int(hold_last * fps)
    colors = ("C1", "C0")
    data = ((ax_a, frames_a, ta, la, sa, Ta, labels[0], colors[0]),
            (ax_b, frames_b, tb, lb, sb, Tb, labels[1], colors[1]))
    with writer.saving(fig, fname, dpi=dpi):
        for tau in taus:
            for ax in (ax_a, ax_b, ax2):
                ax.clear()
            for ax, fr, tt, ll, sw, Tt, lab, col in data:
                k = max(0, int(np.searchsorted(tt, tau, side="right")) - 1)
                f = fr[k]
                draw_hole(ax, f["p"], f["t"], f["hole"])
                in_flow = sw is not None and k >= sw
                phase = "Alg. 3 (shape flow)" if in_flow else lab.split(" ")[0] + " " + lab.split(" ")[1]
                it = k - sw if in_flow else k
                done = tau >= Tt
                status = f"finished in {Tt:.1f} s" if done else f"t = {tau:.1f} s"
                ax.set_title(f"{lab}   [{status}]\n{phase}, iteration {it},  $\\lambda$ = {f['lam']:.5f}",
                             fontsize=10, color=col)
                # eigenvalue against computation time (step curve)
                xs = np.append(tt[:k + 1], min(tau, Tt))
                ys = np.append(ll[:k + 1], ll[k])
                ax2.step(xs, ys, where="post", color=col, lw=1.6, label=lab)
                ax2.plot([xs[-1]], [ys[-1]], "o", color=col, ms=5)
                if in_flow:
                    ax2.axvline(tt[sw], color=col, ls=":", lw=1.2)
                if done:
                    ax2.annotate(f"{Tt:.1f} s", (Tt, ll[k]), textcoords="offset points", xytext=(4, 6),
                                 color=col, fontsize=9)
            ax2.set_xlim(0, T * 1.08)
            ax2.set_ylim(lo - pad, hi + pad)
            ax2.set_xlabel("computation time (s)")
            ax2.set_ylabel(r"$\lambda_{1,h}$")
            ax2.set_title("eigenvalue vs. computation time (dotted line: switch to Alg. 3)", fontsize=10)
            ax2.grid(alpha=0.3)
            ax2.legend(loc="upper right", fontsize=9)
            fig.suptitle(f"{title}   (normalized time: 1 s of movie = {T / duration:.2g} s of computation)",
                         fontsize=11)
            writer.grab_frame()
    plt.close(fig)
