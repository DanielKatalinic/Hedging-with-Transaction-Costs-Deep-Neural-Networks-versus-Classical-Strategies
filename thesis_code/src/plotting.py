"""Drawing code for the thesis figures. Every plot_* function takes plain arrays and returns a figure.

save_figure(fig, name) writes the figure as vector PDF and PNG (dpi 200). The thesis style is set
once for the whole module with plt.rcParams.update(THESIS_STYLE).
"""

import logging
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, PowerNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from src.benchmarks import delta_neutral_line
from src.config import RISK_AVERSION
from src.evaluation import rho_lambda

# fontTools warns about the fixed PDF creation date used for reproducible files
logging.getLogger("fontTools").setLevel(logging.ERROR)

THESIS_STYLE = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,  # embedded TrueType fonts instead of Type 3
}
plt.rcParams.update(THESIS_STYLE)  # applies to every figure drawn and saved from here on

# A4 text width in inches; change this instead of scaling in LaTeX so the font stays at 9 pt
FIGURE_WIDTH = 6.3

# One colour per strategy in every figure (Okabe-Ito, colour-blind safe)
COLORS = {
    "NN": "#0072B2",
    "Rothert": "#E69F00",
    "Gamma-neutral": "#009E73",
    "BS delta": "#D55E00",
    "Reference": "#000000",
}
# Markers as a greyscale backup
MARKERS = {"NN": "X", "Rothert": "s", "Gamma-neutral": "o"}

HEIGHT_DELTA = 3.0
HEIGHT_POSITION_SPACE = 4.6
HEIGHT_MEAN_POSITIONS = 3.4
HEIGHT_STOCK_HEDGE = 4.2
HEIGHT_HOURGLASS = 6.6
HEIGHT_PNL = 3.2
HEIGHT_SWEEP = 3.1
HEIGHT_HEATMAP = 3.9
HEIGHT_ROBUSTNESS = 3.4
ROBUSTNESS_LEGEND_PT = 7
BIN_WIDTH = 0.125
POSITION_PADDING = 0.15
POSITION_LINE_POINTS = 200
HOURGLASS_SPAN = 1.16
HEATMAP_POWERNORM_GAMMA = 0.5
HEATMAP_BOUNDARY_POINTS = 401
PNL_STRATEGIES = ("NN", "Rothert", "Gamma-neutral")


def save_figure(fig, name, out_dir="figures"):
    """Write name.pdf and name.png (dpi 200) to out_dir and close the figure.

    No tight bounding box: the width must stay exactly FIGURE_WIDTH. Returns the two paths.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, metadata={"CreationDate": None})
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"saved {pdf_path} and {png_path.name}")
    return pdf_path, png_path


def kappa_label(kappa):
    """Mathtext 'kappa = 10^n' for a power of ten; raises ValueError otherwise."""
    exponent = round(math.log10(kappa))
    if not math.isclose(kappa, 10.0 ** exponent, rel_tol=1e-9):
        raise ValueError(f"kappa={kappa!r} is not a power of ten")
    return rf"$\kappa = 10^{{{exponent}}}$"


def histogram_edges(window, width):
    """Class edges lo ... hi in steps of width."""
    lo, hi = window
    n = int(round((hi - lo) / width))
    return np.linspace(lo, hi, n + 1)


def density_weights(values, width):
    """Bar weights 1 / (n * width): the bar height is the density relative to all n paths."""
    n = len(values)
    return np.full(n, 1.0 / (n * width))


def _pnl_histograms(ax, pnl, edges, legend_pt):
    """Overlaid density histograms (widest first) with std and rho in the legend. Returns the highest bar.

    pnl: dict {strategy: terminal P&L array}.
    """
    handles = {}
    heights = []
    for name in reversed(PNL_STRATEGIES):
        values = np.asarray(pnl[name], dtype=np.float64)
        n, _, bars = ax.hist(
            values, bins=edges, weights=density_weights(values, BIN_WIDTH),
            color=COLORS[name], alpha=0.5,
        )
        handles[name] = bars
        heights.append(float(n.max()))
    ax.axvline(0.0, color=COLORS["Reference"], linewidth=0.6, linestyle="--")
    ax.set_xlim(edges[0], edges[-1])
    texts = []
    for name in PNL_STRATEGIES:
        values = np.asarray(pnl[name], dtype=np.float64)
        texts.append(
            rf"{name} (std {values.std():.3f}, "
            rf"$\rho_a$ {rho_lambda(pnl[name], RISK_AVERSION):.3f})"
        )
    # opaque legend box: the dashed zero line would run through the text
    ax.legend(
        [handles[name] for name in PNL_STRATEGIES], texts, loc="upper left",
        frameon=True, facecolor="white", framealpha=1.0, edgecolor="none",
        fontsize=legend_pt,
    )
    return max(heights)


def plot_pnl_with_costs(pnl_nn, pnl_rothert, pnl_gn, window):
    """Terminal P&L of NN, Rothert and Gamma-neutral hedge with costs, one panel.

    window = (lo, hi): drawn P&L range; paths outside are not drawn.
    """
    pnl = {"NN": pnl_nn, "Rothert": pnl_rothert, "Gamma-neutral": pnl_gn}
    edges = histogram_edges(window, BIN_WIDTH)
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH, HEIGHT_PNL), layout="constrained"
    )
    peak = _pnl_histograms(ax, pnl, edges, 8)
    ax.set_ylim(0.0, 1.55 * peak)
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Density")
    return fig


def plot_delta(s_grid, delta_nn, delta_bs):
    """NN delta against Black-Scholes delta over the stock price at one time step."""
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH, HEIGHT_DELTA), layout="constrained"
    )
    ax.plot(
        s_grid, delta_bs, color=COLORS["BS delta"],
        linestyle="-", label="Black-Scholes delta",
    )
    ax.plot(
        s_grid, delta_nn, color=COLORS["NN"],
        linestyle="--", label="NN delta",
    )
    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Delta")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    return fig


def plot_position_space(positions):
    """Positions (phi_1, phi_2) of NN and Gamma-neutral hedge on one path at several steps.

    Delta-neutral lines are grey by time step; the axis range is the point range plus a padding.
    """
    fig, ax = plt.subplots(
        figsize=(FIGURE_WIDTH, HEIGHT_POSITION_SPACE), layout="constrained"
    )
    all_s = np.concatenate([positions["nn_stock"].values, positions["gn_stock"].values])
    all_c = np.concatenate([positions["nn_call"].values, positions["gn_call"].values])
    s_min = float(all_s.min()) - POSITION_PADDING
    s_max = float(all_s.max()) + POSITION_PADDING
    c_min = float(all_c.min()) - POSITION_PADDING
    c_max = float(all_c.max()) + POSITION_PADDING
    s_grid = np.linspace(s_min, s_max, POSITION_LINE_POINTS)

    steps = positions["step"].values
    cmap = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.35, 0.9, 256)))
    norm = Normalize(vmin=float(steps.min()), vmax=float(steps.max()))

    for _, row in positions.iterrows():
        line = delta_neutral_line(row["delta_target"], row["delta_hedge"], s_grid)
        ax.plot(
            s_grid, line, color=cmap(norm(float(row["step"]))), alpha=0.6,
            linewidth=1.0, zorder=1,
        )

    # time course first, so the markers lie on top
    ax.plot(
        positions["gn_stock"], positions["gn_call"], color=COLORS["Gamma-neutral"],
        linewidth=1.0, linestyle="--", alpha=0.6, zorder=2,
    )
    ax.plot(
        positions["nn_stock"], positions["nn_call"], color=COLORS["NN"],
        linewidth=1.0, linestyle="-", alpha=0.6, zorder=2,
    )
    ax.scatter(
        positions["gn_stock"], positions["gn_call"],
        marker=MARKERS["Gamma-neutral"], s=70, color=COLORS["Gamma-neutral"],
        edgecolors="black", linewidths=0.7, zorder=4,
    )
    ax.scatter(
        positions["nn_stock"], positions["nn_call"],
        marker=MARKERS["NN"], s=90, color=COLORS["NN"],
        edgecolors="black", linewidths=0.7, zorder=5,
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Time step k"
    )
    legend = ax.legend(
        handles=[
            Line2D([0], [0], color="gray", alpha=0.6, linewidth=1.0,
                   label="Delta-neutral lines"),
            Line2D([0], [0], marker=MARKERS["Gamma-neutral"], color="w",
                   markerfacecolor=COLORS["Gamma-neutral"], markeredgecolor="black",
                   markersize=8, linewidth=0, label="Gamma-neutral position"),
            Line2D([0], [0], marker=MARKERS["NN"], color="w",
                   markerfacecolor=COLORS["NN"], markeredgecolor="black",
                   markersize=9, linewidth=0, label="NN position"),
            Line2D([0], [0], color=COLORS["NN"], linewidth=1.0, alpha=0.6,
                   label="NN path over time"),
            Line2D([0], [0], color=COLORS["Gamma-neutral"], linewidth=1.0,
                   linestyle="--", alpha=0.6, label="Gamma-neutral path over time"),
        ],
        fontsize=8, loc="upper right",
    )
    ax.set_xlabel(r"$\phi_1$ (stock position)")
    ax.set_ylabel(r"$\phi_2$ (call position)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(s_min, s_max)
    ax.set_ylim(c_min, c_max)
    _place_time_labels(fig, ax, positions, legend)
    return fig


def plot_mean_positions(stock_nn, call_nn, stock_gn, call_gn, band=(0.1, 0.9)):
    """Mean positions of NN and Gamma-neutral hedge over the time steps: (a) phi_1, (b) phi_2.

    Positions have shape (n_paths, n_steps). The shaded areas are the band = (lo, hi) quantiles
    over the paths at each step, so the panels show the spread around the mean as well.
    """
    steps = np.arange(np.shape(stock_nn)[1])
    panels = (
        ("(a) Stock position", r"$\phi_1$ (stock position)",
         {"Gamma-neutral": stock_gn, "NN": stock_nn}),
        ("(b) Call position", r"$\phi_2$ (call position)",
         {"Gamma-neutral": call_gn, "NN": call_nn}),
    )
    styles = {"NN": "-", "Gamma-neutral": "--"}
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, HEIGHT_MEAN_POSITIONS), sharex=True, layout="constrained"
    )
    for ax, (title, ylabel, positions) in zip(axes, panels):
        # Gamma-neutral first, so the NN mean lies on top where the two coincide
        for name, values in positions.items():
            values = np.asarray(values, dtype=np.float64)
            lo, hi = np.quantile(values, band, axis=0)
            ax.fill_between(steps, lo, hi, color=COLORS[name], alpha=0.15, linewidth=0)
            ax.plot(
                steps, values.mean(axis=0), color=COLORS[name], linestyle=styles[name],
                linewidth=1.4, marker=MARKERS[name], markersize=4, markevery=5,
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Time step k")
        ax.set_ylabel(ylabel)
        ax.set_xlim(steps[0], steps[-1])
        ax.grid(True, alpha=0.3)

    band_label = f"{100 * band[0]:.0f}-{100 * band[1]:.0f}% quantile band"
    fig.legend(
        handles=[
            Line2D([0], [0], color=COLORS["NN"], linestyle=styles["NN"], linewidth=1.4,
                   marker=MARKERS["NN"], markersize=4, label="NN mean"),
            Line2D([0], [0], color=COLORS["Gamma-neutral"], linestyle=styles["Gamma-neutral"],
                   linewidth=1.4, marker=MARKERS["Gamma-neutral"], markersize=4,
                   label="Gamma-neutral mean"),
            Patch(facecolor=to_rgba(COLORS["NN"], 0.15), label=f"NN {band_label}"),
            Patch(facecolor=to_rgba(COLORS["Gamma-neutral"], 0.15),
                  label=f"Gamma-neutral {band_label}"),
        ],
        fontsize=8, loc="outside lower center", ncol=2,
    )
    return fig


def _overlaps(a, b):
    """True if the rectangles (x0, y0, x1, y1) intersect."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_time_labels(fig, ax, positions, legend, fontsize=8):
    """Label the NN points with 'k=...'.

    Each label goes to the first candidate offset (right, left, then the four corners) with the
    fewest overlaps with markers, other labels, the legend and the axis border. Works in points
    after the final layout.
    """
    fig.draw_without_rendering()
    factor = 72.0 / fig.dpi

    def _pt(s, c):
        return np.asarray(ax.transData.transform((s, c))) * factor

    points = [
        _pt(s, c)
        for col_s, col_c in (("nn_stock", "nn_call"), ("gn_stock", "gn_call"))
        for s, c in zip(positions[col_s], positions[col_c])
    ]
    half = 6.0  # half marker size in pt
    occupied = [tuple(np.asarray(legend.get_window_extent(fig.canvas.get_renderer())
                                 .extents) * factor)]
    border = tuple(np.asarray(ax.bbox.extents) * factor)
    candidates = (
        ((9, 0), "left", "center"), ((-9, 0), "right", "center"),
        ((7, 7), "left", "bottom"), ((-7, 7), "right", "bottom"),
        ((7, -7), "left", "top"), ((-7, -7), "right", "top"),
    )
    for _, row in positions.iterrows():
        p = _pt(row["nn_stock"], row["nn_call"])
        text = f"k={int(row['step'])}"
        width, height = 0.6 * fontsize * len(text), 1.2 * fontsize
        best = None
        for (dx, dy), ha, va in candidates:
            x0 = p[0] + dx - (width if ha == "right" else 0.0)
            y0 = p[1] + dy - {"top": height, "center": 0.5 * height, "bottom": 0.0}[va]
            rect = (x0, y0, x0 + width, y0 + height)
            cost = sum(
                _overlaps(rect, (q[0] - half, q[1] - half, q[0] + half, q[1] + half))
                for q in points
            ) + sum(_overlaps(rect, b) for b in occupied)
            if not (border[0] <= x0 and x0 + width <= border[2]
                    and border[1] <= y0 and y0 + height <= border[3]):
                cost += 10
            if best is None or cost < best[0]:
                best = (cost, rect, (dx, dy), ha, va)
        occupied.append(best[1])
        ax.annotate(
            text, (row["nn_stock"], row["nn_call"]), textcoords="offset points",
            xytext=best[2], ha=best[3], va=best[4], fontsize=fontsize,
        )


def plot_stock_hedge_costs(pnl, costs, ss, dd, zz, step):
    """One-asset network with costs: (a) terminal P&L, (b) costs per path, (c) decision surface."""
    # manual margins instead of constrained layout, which does not suit 3D axes
    fig = plt.figure(figsize=(FIGURE_WIDTH, HEIGHT_STOCK_HEDGE))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.25], left=0.085, right=0.955, bottom=0.115,
        top=0.94, wspace=0.14, hspace=0.62,
    )
    panels = (
        (gs[0, 0], pnl, "(a) Terminal P&L", "Terminal P&L", 3),
        (gs[1, 0], costs, "(b) Transaction costs per path",
         "Transaction costs", 4),
    )
    for cell, values, title, xlabel, digits in panels:
        ax = fig.add_subplot(cell)
        heights, _, _ = ax.hist(
            values, bins=50, density=True, color=COLORS["NN"], alpha=0.7
        )
        # room for the legend above the bars
        ax.set_ylim(0.0, 1.45 * float(heights.max()))
        ax.axvline(
            values.mean(), color=COLORS["Reference"], linestyle="--", linewidth=1.0,
            label=f"mean {values.mean():.{digits}f}",
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper left")

    ax3 = fig.add_subplot(gs[:, 1], projection="3d")
    ax3.plot_surface(ss, dd, zz, cmap="viridis", alpha=0.85)
    ax3.set_xlabel("Stock price", labelpad=2)
    ax3.set_ylabel("Previous delta", labelpad=2)
    ax3.set_zlabel("New delta", labelpad=2)
    ax3.set_box_aspect((1, 1, 0.8), zoom=0.86)
    ax3.tick_params(pad=0)
    ax3.set_title(f"(c) Delta decision at k = {step}", loc="left")
    return fig


def _hourglass_axis_range(values):
    """Midpoint and extent of a coordinate range."""
    lo, hi = float(np.min(values)), float(np.max(values))
    return 0.5 * (lo + hi), hi - lo


def plot_hourglass(panels):
    """2x2 panels, one trade step each: region, positions before (hollow) and after (filled) the trade."""
    fig, axes = plt.subplots(
        2, 2, figsize=(FIGURE_WIDTH, HEIGHT_HOURGLASS), layout="constrained",
        squeeze=False,
    )
    rothert_color = COLORS["Rothert"]
    nn_color = COLORS["NN"]
    for i, panel in enumerate(panels):
        ax = axes[i // 2, i % 2]
        all_s = np.concatenate([
            panel["boundary_stock"],
            [panel["rothert_before"][0], panel["rothert_after"][0], panel["nn_before"][0],
             panel["nn_after"][0], panel["midpoint"][0]],
        ])
        all_c = np.concatenate([
            panel["boundary_call"],
            [panel["rothert_before"][1], panel["rothert_after"][1], panel["nn_before"][1],
             panel["nn_after"][1], panel["midpoint"][1]],
        ])
        mid_s, span_s = _hourglass_axis_range(all_s)
        mid_c, span_c = _hourglass_axis_range(all_c)
        half = 0.5 * (HOURGLASS_SPAN * max(span_s, span_c) + 1e-6)

        ax.fill(
            panel["boundary_stock"], panel["boundary_call"], facecolor=to_rgba(rothert_color, 0.2),
            edgecolor=rothert_color, linewidth=1.0, zorder=1,
        )
        ax.annotate(
            "", xy=panel["rothert_after"], xytext=panel["rothert_before"],
            arrowprops={"arrowstyle": "->", "color": rothert_color, "linewidth": 1.2,
                        "linestyle": "-", "shrinkA": 0, "shrinkB": 0},
            zorder=2,
        )
        ax.annotate(
            "", xy=panel["nn_after"], xytext=panel["nn_before"],
            arrowprops={"arrowstyle": "->", "color": nn_color, "linewidth": 1.2,
                        "linestyle": "--", "shrinkA": 0, "shrinkB": 0},
            zorder=2,
        )
        ax.scatter(
            *panel["rothert_before"], marker=MARKERS["Rothert"], s=45,
            facecolors="none", edgecolors=rothert_color, linewidths=1.3, zorder=3,
        )
        ax.scatter(
            *panel["rothert_after"], marker=MARKERS["Rothert"], s=45,
            color=rothert_color, edgecolors="black", linewidths=0.7, zorder=4,
        )
        ax.scatter(
            *panel["nn_before"], marker=MARKERS["NN"], s=55,
            facecolors="none", edgecolors=nn_color, linewidths=1.3, zorder=3,
        )
        ax.scatter(
            *panel["nn_after"], marker=MARKERS["NN"], s=55,
            color=nn_color, edgecolors="black", linewidths=0.7, zorder=4,
        )
        # small filled circle on top, so a nearly coincident position stays visible
        ax.plot(
            *panel["midpoint"], marker=MARKERS["Gamma-neutral"],
            markerfacecolor=COLORS["Gamma-neutral"], markeredgecolor="black",
            markeredgewidth=0.6, markersize=5, linestyle="none", zorder=5,
        )
        ax.set_xlim(mid_s - half, mid_s + half)
        ax.set_ylim(mid_c - half, mid_c + half)
        ax.set_aspect("equal")
        ax.set_title(f"({'abcd'[i]}) k = {panel['k']}", loc="left")
        ax.grid(True, alpha=0.25)
        if i // 2 == 1:
            ax.set_xlabel(r"$\phi_1$ (stock position)")
        if i % 2 == 0:
            ax.set_ylabel(r"$\phi_2$ (call position)")

    handles = [
        Line2D([0], [0], marker=MARKERS["Rothert"], linestyle="none",
               markerfacecolor="none", markeredgecolor=rothert_color,
               markeredgewidth=1.3, markersize=6,
               label="Rothert: position before trade"),
        Line2D([0], [0], marker=MARKERS["Rothert"], linestyle="none",
               markerfacecolor=rothert_color, markeredgecolor="black", markersize=6,
               label="Rothert: position after trade"),
        Line2D([0], [0], marker=MARKERS["NN"], linestyle="none",
               markerfacecolor="none", markeredgecolor=nn_color,
               markeredgewidth=1.3, markersize=7, label="NN: position before trade"),
        Line2D([0], [0], marker=MARKERS["NN"], linestyle="none",
               markerfacecolor=nn_color, markeredgecolor="black", markersize=7,
               label="NN: position after trade"),
        Line2D([0], [0], color=rothert_color, linewidth=1.2, label="Rothert move"),
        Line2D([0], [0], color=nn_color, linewidth=1.2, linestyle="--",
               label="NN move"),
        Patch(facecolor=to_rgba(rothert_color, 0.2), edgecolor=rothert_color,
              label="Hourglass region"),
        Line2D([0], [0], marker=MARKERS["Gamma-neutral"], linestyle="none",
               markerfacecolor=COLORS["Gamma-neutral"], markeredgecolor="black",
               markeredgewidth=0.6, markersize=5,
               label="Gamma-neutral point (region midpoint)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8)
    return fig


def plot_heatmap(kappas, maps_nn, maps_rothert, edges, color_max):
    """Mean normalised parallel adjustment |x_after - x_before| per cell of the
    normalised position plane, NN and Rothert x kappa.

    maps_nn, maps_rothert: one map per kappa (index [x cell, y cell]); edges: cell edges of the maps.
    A white cell has fewer observations than the minimum count, it does not mean 'no trade'. The
    colour scale ends at color_max; the arrow on the colour bar marks the clip.
    """
    strategies = ("NN", "Rothert")
    maps = {"NN": maps_nn, "Rothert": maps_rothert}

    fig, axes = plt.subplots(
        2, len(kappas), figsize=(FIGURE_WIDTH, HEIGHT_HEATMAP), sharex=True, sharey=True,
        squeeze=False, layout="constrained",
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    norm = PowerNorm(
        gamma=HEATMAP_POWERNORM_GAMMA, vmin=0.0, vmax=color_max
    )

    # boundary of the hourglass |y| = |x|^(2/3), |x| <= 1, as a closed polyline
    xs = np.linspace(-1.0, 1.0, HEATMAP_BOUNDARY_POINTS)
    upper = np.abs(xs) ** (2.0 / 3.0)
    edge_x = np.concatenate([xs, xs[::-1], xs[:1]])
    edge_y = np.concatenate([upper, -upper[::-1], upper[:1]])

    im = None
    for i, strategy in enumerate(strategies):
        for j, kappa in enumerate(kappas):
            ax = axes[i, j]
            # maps are indexed [x cell, y cell], hence the transpose for pcolormesh
            im = ax.pcolormesh(
                edges, edges, maps[strategy][j].T,
                cmap=cmap, norm=norm, shading="flat",
            )
            ax.plot(edge_x, edge_y, color=COLORS["Rothert"], linewidth=1.0)
            ax.set_title(f"{strategy}, {kappa_label(kappa)}")
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_aspect("equal")
            if i == 1:
                ax.set_xlabel(r"$x = d^{\parallel} / h^{\parallel}$")
            if j == 0:
                ax.set_ylabel(r"$y = d^{\perp} / h^{\perp}(h^{\parallel})$")

    fig.colorbar(
        im, ax=axes, shrink=0.85, extend="max",
        label="mean normalised parallel adjustment\n" + r"$|x_{\mathrm{after}} - x_{\mathrm{before}}|$",
    )
    return fig


def plot_kappa_sweep(kappas, rho_nn, rho_rothert, rho_gn):
    """(a) rho_a over kappa for the three strategies, (b) ratio rho(NN) / rho(Rothert).

    kappas and the three rho lists have the same length and order; any order of kappas works.
    """
    kappas = np.asarray(kappas, dtype=np.float64)
    rho = {
        "NN": np.asarray(rho_nn, dtype=np.float64),
        "Rothert": np.asarray(rho_rothert, dtype=np.float64),
        "Gamma-neutral": np.asarray(rho_gn, dtype=np.float64),
    }
    ascending = np.argsort(kappas)  # panel (a) is drawn from small to large kappa
    descending = ascending[::-1]  # panel (b) from large to small kappa

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, HEIGHT_SWEEP), layout="constrained"
    )
    for name in PNL_STRATEGIES:
        ax1.plot(
            kappas[ascending], rho[name][ascending], marker=MARKERS[name],
            markersize=5, linewidth=1.2, color=COLORS[name], label=name,
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title(r"(a) $\rho_a$ vs. $\kappa$", loc="left")
    ax1.set_xlabel(r"$\kappa$")
    ax1.set_ylabel(r"$\rho_a$ (lower is better)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(
        kappas[descending], rho["NN"][descending] / rho["Rothert"][descending],
        marker="o", markersize=5, linewidth=1.2, color=COLORS["Reference"],
    )
    ax2.axhline(
        1.0, color=COLORS["Reference"], linestyle="--", linewidth=0.8,
        label="ratio = 1",
    )
    ax2.set_xscale("log")
    ax2.set_title(r"(b) $\rho_a(\mathrm{NN}) / \rho_a(\mathrm{Rothert})$",
                  loc="left")
    ax2.set_xlabel(r"$\kappa$")
    ax2.set_ylabel("Ratio")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)
    return fig


def plot_robustness(left_nn, left_rothert, left_gn, right_nn, right_rothert, right_gn, titles, window):
    """1x2 panels of terminal P&L histograms sharing x axis, classes and y axis.

    titles: the two panel titles; window = (lo, hi): drawn P&L range, paths outside are not drawn.
    """
    panels = (
        {"NN": left_nn, "Rothert": left_rothert, "Gamma-neutral": left_gn},
        {"NN": right_nn, "Rothert": right_rothert, "Gamma-neutral": right_gn},
    )
    edges = histogram_edges(window, BIN_WIDTH)
    fig, axes = plt.subplots(
        1, 2, figsize=(FIGURE_WIDTH, HEIGHT_ROBUSTNESS), sharex=True, sharey=True,
        squeeze=False, layout="constrained",
    )
    peaks = []
    for ax, pnl, title in zip(axes[0], panels, titles):
        peaks.append(_pnl_histograms(ax, pnl, edges, ROBUSTNESS_LEGEND_PT))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Terminal P&L")
    axes[0, 0].set_ylim(0.0, 1.55 * max(peaks))
    axes[0, 0].set_ylabel("Density")
    return fig
