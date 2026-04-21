"""
Equilibrium region plot in (X, Y) space.
Three regimes: LL (red), LH (blue), HH (green) — no sub-regime distinction.
Fixed axes: X in [0, 1.5], Y in [0, 0.15].
Call with different cS to get different files.
"""

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Cool light colour scheme (matching casestudy style) ───────────────────────
C_LL = np.array([0.99, 0.82, 0.82])   # light pink-red
C_LH = np.array([0.87, 0.92, 0.97])   # light blue  (#DEEBF7)
C_HH = np.array([0.82, 0.96, 0.86])   # light green

L_LL = "#C0392B"   # boundary line colours
L_LH = "#2166AC"
L_HH = "#2C6E49"

def make_plot(cS, outfile):
    # ── Fixed model parameters ─────────────────────────────────────────────
    b = 0.8; H = 1.0; L = 0.8
    Lb = 1-L; HL = H*L; rhoR = 0.8; rhoS = 0.8
    b2 = b*b; bm = 2-b; bp = 1+b; cb = 1-cS; L2 = L*L

    # ── Scalar thresholds ──────────────────────────────────────────────────
    X_LL   = 1/(2*rhoR*bm**2*bp*(1-L2))
    Y_LL   = cS*(b*(cS-4)-2*cS+4)/(8*bm*(1-b)*bp*rhoS*(H-L))

    alpha  = cb/bm; A_HH=(2*(1-b)+b*cS)/bm; U_HH=(1-b)*cb**2/(2*bp*bm**2)
    X_HH_d = (b2*cS*(4-3*cS)+4*b*(cS-2)+4*(cS-2)*cS+8)/(16*bm**2*(b2-1)*rhoR*(HL-1))
    X_HH_D = cb**2/(4*b2*bm**2*rhoR*(1-HL))
    Y_HH_0 = cS*(4*(1-b)+cS*(3*b-2))/(8*bm*(1-b2)*rhoS*Lb)

    # LH-hld outer lower bound Y^{LHhld-LB-LLFOC}  (from equilibriaandbounds_v2.tex)
    # Valid for X ≤ bar X^{LL}; marks the LOWEST Y where any LH equilibrium exists.
    _inum = ((b-1)**2*b**3 - 2*(b-2)**2*(b2+b-2)*cS + (b-2)**2*b*cS**2)
    _iden = (b-2)**2*b*cS**2
    _inner = (b2-1)**2*rhoS**2*(H-L)**2 * _inum/_iden
    if _inner >= 0:
        _sq = np.sqrt(_inner)
        _T1 = 2*(b2-1)**2*cS*rhoS*(H-L)
        _T2 = b2*cS*_sq
        _T3 = bp*((b-2)*b-4)*(b-1)**2*rhoS*(H-L)
        Y_LH_LB = (_T1+_T2+_T3)*cS / (8*(b2-2)*(b2-1)**2*rhoS**2*(H-L)**2)
    else:
        Y_LH_LB = 0.0   # fallback

    print(f"cS={cS}  X_LL={X_LL:.3f}  Y_LL={Y_LL:.4f}")
    print(f"       Y_LH_LB={Y_LH_LB:.4f}  overlap_width={Y_LL-Y_LH_LB:.4f}")
    print(f"       X_HH_d={X_HH_d:.3f}  X_HH_D={X_HH_D:.3f}  Y_HH_0={Y_HH_0:.4f}")

    # ── Boundary helpers ───────────────────────────────────────────────────
    def hh_lb(X):
        if X > X_HH_D: return 0.0
        if X > X_HH_d:
            disc = 4*X*rhoR*Lb - alpha**2
            if disc < 0: return Y_HH_0
            D = np.sqrt(disc)
            return max((A_HH*D/(2*np.sqrt(1-b2))-D**2/2-U_HH)/(rhoS*Lb), 0.0)
        return Y_HH_0

    def ll_sqz_ub(X):
        T2 = bp*(1-L2)*rhoR*X
        if T2 <= 0: return 0.0
        T = np.sqrt(T2)
        num = (2*np.sqrt(2)*b*cS*T - 2*bp*bm**2*(L2-1)*rhoR*X
               + 2*np.sqrt(2)*b*T - 4*np.sqrt(2)*T + (cS-1)**2)
        return num/(8*(b2-1)*rhoS*(H-L))

    def ll_ub(X):
        # bar Y^{LL}(X): outer upper bound of LL regime
        return Y_LL if X <= X_LL else max(ll_sqz_ub(X), 0.0)

    def lh_lb(X):
        # Outer lower bound of LH regime:
        #   X ≤ bar X^{LL}: Y^{LHhld-LB-LLFOC} (constant, from tex formula)
        #   X >  bar X^{LL}: 0  (LH-sqz/hld fills entire [0, Y_HH) gap)
        return Y_LH_LB if X <= X_LL else 0.0

    # ── Dynamic axis limits ────────────────────────────────────────────────
    Xmax = 1.5
    Ymax = max(Y_HH_0, Y_LL) * 1.55
    Ymax = round(Ymax * 20) / 20   # round to nearest 0.05

    # ── Grid ──────────────────────────────────────────────────────────────
    NX, NY = 700, 700
    Xg = np.linspace(0, Xmax, NX)
    Yg = np.linspace(0, Ymax, NY)
    XX, YY = np.meshgrid(Xg, Yg)

    ll_ub_arr = np.array([ll_ub(xv) for xv in Xg])
    hh_lb_arr = np.array([hh_lb(xv) for xv in Xg])
    lh_lb_arr = np.array([lh_lb(xv) for xv in Xg])

    # ── Independent regime masks (overlaps allowed) ────────────────────────
    # LL: Y ≤ bar Y^{LL}(X)
    LL = YY <= ll_ub_arr[np.newaxis, :]

    # HH: Y ≥ check Y^{HH}(X)
    HH = YY >= hh_lb_arr[np.newaxis, :]

    # LH: Y^{LH,lb}(X) ≤ Y < check Y^{HH}(X)
    #   Lower bound from Y^{LHhld-LB-LLFOC} (for X ≤ bar X^{LL}) and 0 thereafter.
    #   Upper bound is strictly below HH (LH-hld requires Y < check Y^{HH}).
    LH = (YY >= lh_lb_arr[np.newaxis, :]) & (YY < hh_lb_arr[np.newaxis, :])

    # ── Colour image with overlaps ─────────────────────────────────────────
    r_ch = LL.astype(float)   # LL channel
    b_ch = LH.astype(float)   # LH channel
    g_ch = HH.astype(float)   # HH channel

    # Overlap tint: LL∩LH → redder-blue blend; LL∩HH and LH∩HH absent here
    C_OV = (C_LL + C_LH) / 2   # LL ∩ LH overlap colour

    img = np.ones((NY, NX, 4))
    LL_only  = LL & ~LH & ~HH
    LH_only  = LH & ~LL & ~HH
    HH_only  = HH & ~LL & ~LH
    LL_LH    = LL & LH & ~HH     # the visible overlap region
    nowhere  = ~LL & ~LH & ~HH

    for c in range(3):
        img[:,:,c] = (LL_only *C_LL[c] + LH_only *C_LH[c] +
                      HH_only *C_HH[c] + LL_LH   *C_OV[c] +
                      nowhere * 1.0)
    img[:,:,3] = 1.0

    # ── Figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.imshow(img, origin="lower", extent=[0, Xmax, 0, Ymax],
              aspect="auto", interpolation="bilinear")

    lw_main = 2.2

    Xp = np.linspace(0, Xmax, 1200)

    # ── Outer boundary lines only (no internal sub-regime lines) ──────────

    # LL outer UB: bar Y^{LL}(X)  — continuous piecewise curve
    ll_line = np.clip([ll_ub(x) for x in Xp], 0, Ymax)
    active  = ll_line > 1e-6
    if active.any():
        ax.plot(Xp[active], ll_line[active], color=L_LL, lw=lw_main, zorder=5)

    # LH outer LB: Y^{LH,lb}(X)  — horizontal at Y_LH_LB for X ≤ bar X^{LL}
    if Y_LH_LB > 0 and Y_LH_LB <= Ymax:
        ax.plot([0, X_LL], [Y_LH_LB, Y_LH_LB], color=L_LH, lw=lw_main, zorder=5)

    # HH outer LB: check Y^{HH}(X)
    hh_line = np.clip([hh_lb(x) for x in Xp], 0, Ymax)
    ax.plot(Xp, hh_line, color=L_HH, lw=lw_main, zorder=5)

    # ── Named boundary annotations ─────────────────────────────────────────
    def blabel(x, y, txt, col, dx=0, dy=0):
        if 0 <= x <= Xmax and 0 < y <= Ymax:
            ax.text(x+dx, y+dy, txt, color=col, fontsize=8,
                    ha="center", va="bottom", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    # LL UB label on curve
    blabel(X_LL*0.40, Y_LL, r"$\bar{Y}^{LL}(X)$", L_LL, dy=Ymax*0.015)

    # LH LB label
    if Y_LH_LB > 0 and Y_LH_LB <= Ymax:
        blabel(X_LL*0.55, Y_LH_LB,
               r"$\check{Y}^{LH}$", L_LH, dy=Ymax*0.015)

    # HH LB label — place at mid-X of flat portion
    blabel(X_HH_d*0.45, Y_HH_0,
           r"$\check{Y}^{HH}(X)$", L_HH, dy=Ymax*0.015)

    # ── Right-spine Y-threshold ticks ─────────────────────────────────────
    y_ticks = [(Y_LL,   r"$\bar{Y}^{LL}$",           L_LL),
               (Y_HH_0, r"$\check{Y}^{HH}$",     L_HH)]
    if Y_LH_LB > 0 and Y_LH_LB <= Ymax:
        y_ticks.append((Y_LH_LB, r"$\check{Y}^{LH}$", L_LH))

    # deduplicate ticks that are too close
    y_ticks.sort(key=lambda t: t[0])
    shown = []
    for yv, lbl, col in y_ticks:
        if yv <= Ymax and all(abs(yv-s) > Ymax*0.02 for s in shown):
            ax.plot([Xmax*0.97, Xmax], [yv, yv], color=col, lw=1.0, ls=":")
            ax.text(Xmax*1.01, yv, lbl, color=col, fontsize=8,
                    ha="left", va="center", zorder=7)
            shown.append(yv)

    # ── Top-spine X-threshold ticks ────────────────────────────────────────
    x_ticks = [(X_LL,   r"$\bar{X}^{LL}$",  L_LL),
               (X_HH_d, r"$X^{HH,\dagger}$",L_HH),
               (X_HH_D, r"$\bar{X}^{HH}$",  L_HH)]
    for xv, lbl, col in x_ticks:
        if xv <= Xmax:
            ax.plot([xv, xv], [Ymax*0.96, Ymax], color=col, lw=1.0, ls=":")
            ax.text(xv, Ymax*1.018, lbl, color=col, fontsize=8,
                    ha="center", va="bottom", zorder=7)

    # ── Regime labels ──────────────────────────────────────────────────────
    def rlab(x, y, txt, col):
        if 0 < x < Xmax and 0 < y < Ymax:
            ax.text(x, y, txt, color=col, ha="center", va="center",
                    fontsize=10, fontweight="bold", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))

    # Pure LL
    rlab(X_LL*0.4, Y_LH_LB*0.5, "LL", L_LL)
    # LL ∩ LH overlap
    if Y_LH_LB > 0:
        ov_mid = (Y_LH_LB + Y_LL)/2
        rlab(X_LL*0.4, ov_mid, "LL\n∩ LH", "#7B3F8A")
    # Pure LH (above LL, below HH)
    lh_mid = (Y_LL + Y_HH_0)/2
    rlab(X_HH_d*0.35, min(lh_mid, Ymax*0.75), "LH", L_LH)
    # LH for X > X_LL
    rlab(min(X_LL*1.6, Xmax*0.7), min(Y_HH_0*0.5, Ymax*0.5), "LH", L_LH)
    # HH
    hh_mid = min(Y_HH_0 + (Ymax-Y_HH_0)*0.45, Ymax*0.92)
    if Y_HH_0 < Ymax:
        rlab(X_HH_d*0.45, hh_mid, "HH", L_HH)

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(0, Xmax)
    ax.set_ylim(0, Ymax)
    ax.set_xlabel(r"$X = K^R\theta^R$", fontsize=11)
    ax.set_ylabel(r"$Y = K^S\theta^S$", fontsize=11)
    ax.set_title(rf"Equilibrium regions ($b=0.8,\ c_S={cS}$)", fontsize=12, pad=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────
    C_OV_t = tuple(C_OV)
    legend_elems = [
        Patch(fc=(*C_LL,1), ec=L_LL, lw=0.8, label="LL feasible"),
        Patch(fc=(*C_LH,1), ec=L_LH, lw=0.8, label="LH feasible"),
        Patch(fc=(*C_HH,1), ec=L_HH, lw=0.8, label="HH feasible"),
        Patch(fc=(*C_OV,1), ec="#7B3F8A", lw=0.8, label="LL $\\cap$ LH (overlap)"),
        Line2D([0],[0], color=L_LL, lw=lw_main,
               label=r"$\bar{Y}^{LL}(X)$: LL upper bound"),
        Line2D([0],[0], color=L_LH, lw=lw_main,
               label=r"$\check{Y}^{LH}$: LH lower bound"),
        Line2D([0],[0], color=L_HH, lw=lw_main,
               label=r"$\check{Y}^{HH}(X)$: HH lower bound"),
    ]
    ax.legend(handles=legend_elems, fontsize=7.8, loc="upper right",
              framealpha=0.93, ncol=1)

    fig.tight_layout()
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.show()
    plt.close(fig)


# ── Run both parameter sets ───────────────────────────────────────────────────
base = os.getcwd()
make_plot(cS=0.05, outfile=base + "\eqm_region_plot.png")
make_plot(cS=0.20, outfile=base + "\eqm_region_plot_cS02.png")
