from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FINAL = ROOT / "outputs" / "final_experiment_tables"
EMB = (
    ROOT
    / "outputs"
    / "method_novelty_v1"
    / "seed_1"
    / "full_structured_relgt"
    / "test_group"
    / "query_embeddings.csv"
)
METHOD_NOVELTY = ROOT / "outputs" / "method_novelty_v1"
ROBUSTNESS = ROOT / "outputs" / "final_robustness_relgt_oscl" / "step34_robustness_summary.csv"
PRED = ROOT / "outputs" / "final_error_analysis_relgt_oscl" / "kcgt" / "kcgt_contrastive_predictions.csv"
GATE = ROOT / "outputs" / "final_explainability_relgt_oscl" / "in_protocol_neutral_kg_gate_by_class.csv"
MANAGEMENT = ROOT / "outputs" / "final_management_validation_relgt_oscl" / "management_validation_final.csv"
PHOTO_DIR = ROOT / "paper" / "assets" / "open_source_photos"
PHOTO_FILES = {
    "factory": PHOTO_DIR / "factory_kuka_public_domain.jpg",
    "service": PHOTO_DIR / "service_server_racks_cc0.jpg",
    "city": PHOTO_DIR / "ar_city_skyline_cc0.jpg",
    "vehicle": PHOTO_DIR / "v2x_waymo_cc_by_sa_4.jpg",
}

INK = "#2f2f2f"
LIGHT = "#f5f5f5"
BLUE = "#6f8fb7"
ORANGE = "#b98b5f"
GREEN = "#7f9f7a"
GRAY = "#8c8c8c"


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Times New Roman", "Arial", "DejaVu Sans"],
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def rect(ax, x, y, w, h, label, fc=LIGHT, ec=INK, lw=0.9, fontsize=8.8):
    patch = Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, label=None, off=(0, 0), rad=0.0, lw=0.8, fontsize=6.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=INK,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + off[0],
            (start[1] + end[1]) / 2 + off[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=INK,
        )
    return arr


def mini_mlp(ax, x, y, w, h, label="2-layer MLP"):
    """Compact neural-network glyph for architecture blocks."""
    xs = [x + 0.18 * w, x + 0.50 * w, x + 0.82 * w]
    ys = [
        [y + 0.30 * h, y + 0.70 * h],
        [y + 0.20 * h, y + 0.50 * h, y + 0.80 * h],
        [y + 0.35 * h, y + 0.65 * h],
    ]
    for left, right in zip(range(2), range(1, 3)):
        for ya in ys[left]:
            for yb in ys[right]:
                ax.plot([xs[left], xs[right]], [ya, yb], color="#9aa4aa", lw=0.32, zorder=2)
    for layer, color in zip(range(3), ("#8fb7d8", "#76aeb7", "#817fa8")):
        for yy in ys[layer]:
            ax.scatter([xs[layer]], [yy], s=13, marker="o", c=[color], edgecolors=INK, linewidths=0.35, zorder=3)
    ax.text(x + w / 2, y + h + 0.004, label, ha="center", va="bottom", fontsize=3.35)


def mini_transformer(ax, x, y, w, h, label):
    colors = ("#d9e8f3", "#b8d7e6", "#aaa8cb")
    widths = (0.24, 0.34, 0.24)
    labels = ("Emb.", "Attn.", "Pool")
    xx = x
    boxes = []
    for frac, color, text in zip(widths, colors, labels):
        bw = frac * w
        boxes.append((xx, bw))
        ax.add_patch(FancyBboxPatch((xx, y), bw, h, boxstyle="round,pad=0.001,rounding_size=0.004",
                                    fc=color, ec="#627b8a", lw=0.45))
        ax.text(xx + bw / 2, y + h / 2, text, ha="center", va="center", fontsize=2.75)
        xx += bw + 0.09 * w
    for (xa, wa), (xb, _) in zip(boxes[:-1], boxes[1:]):
        arrow(ax, (xa + wa, y + h / 2), (xb, y + h / 2), lw=0.35)
    ax.text(x + w / 2, y + h + 0.005, label, ha="center", va="bottom", fontsize=3.25)


def mini_rel_gt(ax, x, y, w, h, label):
    positions = [(0.10, 0.50), (0.40, 0.78), (0.40, 0.22), (0.72, 0.68), (0.72, 0.30), (0.94, 0.50)]
    colors = ("#8fb7d8", "#76aeb7", "#aaa8cb", "#8fb7d8", "#76aeb7", "#817fa8")
    edges = [(0,1),(0,2),(1,3),(1,4),(2,3),(2,4),(3,5),(4,5)]
    for a, b in edges:
        xa, ya = positions[a]; xb, yb = positions[b]
        ax.plot([x + xa*w, x + xb*w], [y + ya*h, y + yb*h], color="#8298a4", lw=0.38, zorder=1)
    for (px, py), color in zip(positions, colors):
        ax.scatter([x + px*w], [y + py*h], s=18, marker="o", c=[color], edgecolors="#4f6470", linewidths=0.4, zorder=2)
    ax.text(x + w / 2, y + h + 0.005, label, ha="center", va="bottom", fontsize=3.20)


def elbow_arrow(ax, points, label=None, off=(0, 0), lw=0.65, fontsize=5.0):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, linewidth=lw)
    arr = arrow(ax, points[-2], points[-1], lw=lw, fontsize=fontsize)
    if label:
        x = (points[0][0] + points[-1][0]) / 2 + off[0]
        y = (points[0][1] + points[-1][1]) / 2 + off[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=INK)
    return arr


def icon(ax, kind: str, x: float, y: float, s: float = 0.026) -> None:
    color = "#4a4a4a"
    lw = 0.75
    if kind == "app":
        ax.add_patch(Rectangle((x - s * 0.55, y - s * 0.38), s * 1.1, s * 0.76, fill=False, ec=color, lw=lw))
        ax.plot([x - s * 0.25, x + s * 0.25], [y - s * 0.38, y - s * 0.38], color=color, lw=lw)
        ax.plot([x, x], [y - s * 0.38, y - s * 0.58], color=color, lw=lw)
    elif kind == "scenario":
        ax.add_patch(Circle((x, y + s * 0.08), s * 0.36, fill=False, ec=color, lw=lw))
        ax.plot([x, x], [y - s * 0.28, y - s * 0.62], color=color, lw=lw)
        ax.plot([x - s * 0.5, x + s * 0.5], [y - s * 0.62, y - s * 0.62], color=color, lw=lw)
    elif kind == "qos":
        ax.plot([x - s * 0.55, x - s * 0.18, x + s * 0.05, x + s * 0.55],
                [y - s * 0.1, y - s * 0.1, y + s * 0.42, y + s * 0.42], color=color, lw=lw)
        ax.plot([x - s * 0.6, x + s * 0.6], [y - s * 0.42, y - s * 0.42], color=color, lw=lw)
    elif kind == "slice":
        for k in range(3):
            yy = y + (k - 1) * s * 0.28
            ax.add_patch(Rectangle((x - s * 0.55, yy - s * 0.08), s * 1.1, s * 0.16, fill=False, ec=color, lw=lw))
    elif kind == "function":
        ax.add_patch(Circle((x, y), s * 0.42, fill=False, ec=color, lw=lw))
        for ang in np.linspace(0, 2 * np.pi, 7)[:-1]:
            ax.plot([x + np.cos(ang) * s * 0.42, x + np.cos(ang) * s * 0.62],
                    [y + np.sin(ang) * s * 0.42, y + np.sin(ang) * s * 0.62], color=color, lw=lw)
    elif kind == "resource":
        ax.add_patch(Rectangle((x - s * 0.55, y - s * 0.38), s * 1.1, s * 0.76, fill=False, ec=color, lw=lw))
        ax.plot([x - s * 0.35, x + s * 0.35], [y + s * 0.12, y + s * 0.12], color=color, lw=lw)
        ax.plot([x - s * 0.35, x + s * 0.35], [y - s * 0.12, y - s * 0.12], color=color, lw=lw)
    elif kind == "relation":
        ax.add_patch(Polygon([(x - s * 0.5, y), (x, y + s * 0.42), (x + s * 0.5, y), (x, y - s * 0.42)],
                             fill=False, ec=color, lw=lw))


def kg_node(ax, x, y, w, h, title, elements, fc, kind, fontsize=6.9):
    patch = Rectangle((x, y), w, h, linewidth=0.9, edgecolor=INK, facecolor=fc)
    ax.add_patch(patch)
    icon(ax, kind, x + 0.018, y + h - 0.022, 0.012)
    ax.text(x + w / 2 + 0.006, y + h * 0.70, title, ha="center", va="center", fontsize=fontsize + 0.15)
    ax.text(x + w / 2, y + h * 0.34, elements, ha="center", va="center", fontsize=fontsize)
    return patch


def group_box(ax, x, y, w, h, label, fc="#f8f8f8", ec="#777777", fontsize=6.8):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
        linestyle=(0, (4, 3)),
        alpha=0.72,
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + h - 0.030, label, ha="left", va="center", fontsize=fontsize, color="#333333")
    return patch


def scene_thumb(kind: str, size: int = 72) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.zeros((size, size, 3), dtype=float)
    if kind == "factory":
        img[..., 0] = 0.78 + 0.08 * yy / size
        img[..., 1] = 0.82 + 0.06 * yy / size
        img[..., 2] = 0.86 + 0.04 * yy / size
        for x in (14, 32, 50):
            img[:, max(0, x - 1):x + 1, :] = 0.45
        img[48:62, 8:62, :] = [0.35, 0.38, 0.40]
        img[28:52, 24:30, :] = [0.70, 0.55, 0.38]
        img[22:28, 28:46, :] = [0.84, 0.62, 0.36]
        img[30:36, 43:58, :] = [0.78, 0.48, 0.32]
    elif kind == "city":
        img[..., :] = [0.78, 0.88, 0.94]
        img[48:, :, :] = [0.45, 0.48, 0.50]
        for x, h in [(5, 28), (18, 38), (33, 24), (46, 34), (58, 30)]:
            img[size - h:48, x:x + 9, :] = [0.38, 0.45, 0.52]
            img[size - h + 4:46:8, x + 2:x + 7, :] = [0.92, 0.82, 0.54]
    elif kind == "vehicle":
        img[..., :] = [0.70, 0.84, 0.92]
        img[48:, :, :] = [0.38, 0.42, 0.43]
        img[52:57, 8:62, :] = [0.90, 0.90, 0.86]
        img[37:50, 20:54, :] = [0.55, 0.70, 0.82]
        img[44:56, 12:62, :] = [0.22, 0.30, 0.38]
        img[53:60, 18:28, :] = [0.05, 0.05, 0.05]
        img[53:60, 48:58, :] = [0.05, 0.05, 0.05]
    else:
        img[..., 0] = 0.72 + 0.08 * np.sin(xx / 7)
        img[..., 1] = 0.78 + 0.08 * np.cos(yy / 9)
        img[..., 2] = 0.84
        for x in range(10, 65, 14):
            img[10:60, x:x + 5, :] = [0.42, 0.48, 0.56]
            img[14:55:8, x + 1:x + 4, :] = [0.80, 0.88, 0.92]
    noise = (np.sin(xx * 0.7 + yy * 1.3) + np.cos(xx * 1.7)) * 0.015
    return np.clip(img + noise[..., None], 0, 1)


def journal_photo(kind: str) -> np.ndarray:
    """Load, crop, and lightly desaturate a source photograph for a compact figure card."""
    path = PHOTO_FILES.get(kind)
    if path is None or not path.exists():
        return scene_thumb(kind)

    img = plt.imread(path)
    if img.dtype.kind in "ui":
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = img[..., :3]

    height, width = img.shape[:2]
    scale = 0.85 if kind == "vehicle" else 1.0
    side = max(1, int(min(height, width) * scale))
    focus = {
        "factory": (0.55, 0.52),
        "service": (0.58, 0.52),
        "city": (0.58, 0.52),
        # The lower crop keeps the autonomous vehicle and excludes people
        # visible only along the original photograph's upper boundary.
        "vehicle": (0.46, 0.60),
    }.get(kind, (0.5, 0.5))
    cx, cy = int(width * focus[0]), int(height * focus[1])
    x0 = min(max(0, cx - side // 2), width - side)
    y0 = min(max(0, cy - side // 2), height - side)
    img = img[y0:y0 + side, x0:x0 + side]

    gray = np.sum(img * np.array([0.2126, 0.7152, 0.0722]), axis=2, keepdims=True)
    img = 0.82 * img + 0.18 * gray
    return np.clip(img, 0, 1)


def photo_card(ax, x, y, w, h, label, kind, fontsize=5.3):
    ax.imshow(journal_photo(kind), extent=(x, x + w, y, y + h), zorder=1, aspect="auto")
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0.8, edgecolor=INK, facecolor="none", zorder=2))
    ax.text(x + w / 2, y - 0.012, label, ha="center", va="top", fontsize=fontsize, color="#333333")


def tiny_icon_box(ax, x, y, label, kind, fc="#ffffff"):
    rect(ax, x, y, 0.070, 0.040, label, fc, fontsize=4.8)
    icon(ax, kind, x + 0.010, y + 0.020, 0.010)


def draw_kg_schema_legacy() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    group_box(ax, 0.015, 0.300, 0.270, 0.650, "A. Multi-domain intent sources", "#f4f6f8", fontsize=6.2)
    group_box(ax, 0.300, 0.300, 0.210, 0.650, "B. Normalization and grounding", "#f7f7f2", fontsize=6.2)
    group_box(ax, 0.525, 0.300, 0.275, 0.650, "C. Intent knowledge graph", "#f7f7f2", fontsize=6.2)
    group_box(ax, 0.815, 0.300, 0.170, 0.650, "D. Management actions", "#f6f4f1", fontsize=6.0)
    group_box(ax, 0.015, 0.045, 0.970, 0.215, "E. Closed-loop representation workflow", "#f8f8f8", fontsize=6.2)

    photo_card(ax, 0.040, 0.805, 0.055, 0.055, "factory", "factory", fontsize=4.5)
    photo_card(ax, 0.120, 0.805, 0.055, 0.055, "AR/city", "city", fontsize=4.5)
    photo_card(ax, 0.200, 0.805, 0.055, 0.055, "V2X", "vehicle", fontsize=4.5)
    rect(ax, 0.040, 0.690, 0.215, 0.062, "Intent text\nrobot: dependable low-latency control", "#ffffff", fontsize=4.9)
    rect(ax, 0.040, 0.600, 0.100, 0.052, "QoS tuple\nlat, rel, bw", "#ffffff", fontsize=4.6)
    rect(ax, 0.155, 0.600, 0.100, 0.052, "Context tuple\nsite, mobility", "#ffffff", fontsize=4.6)
    rect(ax, 0.040, 0.500, 0.100, 0.052, "Slice hint\nURLLC/eMBB", "#ffffff", fontsize=4.6)
    rect(ax, 0.155, 0.500, 0.100, 0.052, "Service clue\nedge control", "#ffffff", fontsize=4.6)
    rect(ax, 0.050, 0.370, 0.195, 0.064, "source views\nnatural language + QoS + context", "#eef2f6", fontsize=4.8)

    # The five source fields are parallel inputs.  A shared collector preserves
    # every required dependency without drawing lines through labels or boxes.
    ax.plot([0.1475, 0.1475], [0.690, 0.470], color=INK, linewidth=0.50)
    ax.plot([0.090, 0.090, 0.1475], [0.600, 0.575, 0.575], color=INK, linewidth=0.50)
    ax.plot([0.205, 0.205, 0.1475], [0.600, 0.575, 0.575], color=INK, linewidth=0.50)
    ax.plot([0.090, 0.090, 0.1475], [0.500, 0.475, 0.475], color=INK, linewidth=0.50)
    ax.plot([0.205, 0.205, 0.1475], [0.500, 0.475, 0.475], color=INK, linewidth=0.50)
    arrow(ax, (0.1475, 0.470), (0.1475, 0.434), lw=0.50)

    rect(ax, 0.350, 0.805, 0.110, 0.052, "entity parser", "#ffffff", fontsize=4.8)
    rect(ax, 0.350, 0.720, 0.110, 0.052, "QoS normalizer", "#ffffff", fontsize=4.8)
    rect(ax, 0.350, 0.635, 0.110, 0.052, "ontology linker", "#ffffff", fontsize=4.8)
    rect(ax, 0.350, 0.550, 0.110, 0.052, "subgraph sampler", "#ffffff", fontsize=4.8)
    rect(ax, 0.325, 0.455, 0.160, 0.060, "intent instance\n$(T_i,\\mathbf{q}_i,\\mathcal{G}_i,y_i)$", "#eef2f6", fontsize=5.0)
    rect(ax, 0.325, 0.345, 0.070, 0.052, "KG\nschema", "#ffffff", fontsize=4.8)
    rect(ax, 0.415, 0.345, 0.070, 0.052, "quality\nchecks", "#ffffff", fontsize=4.8)

    # A single aggregate input keeps the cross-panel routing legible.
    elbow_arrow(ax, [(0.245, 0.402), (0.290, 0.402), (0.290, 0.831), (0.350, 0.831)], lw=0.55)
    arrow(ax, (0.405, 0.805), (0.405, 0.772), lw=0.50)
    arrow(ax, (0.405, 0.720), (0.405, 0.687), lw=0.50)
    arrow(ax, (0.405, 0.635), (0.405, 0.602), lw=0.50)
    arrow(ax, (0.405, 0.550), (0.405, 0.515), lw=0.50)
    arrow(ax, (0.365, 0.455), (0.360, 0.397), lw=0.50)
    arrow(ax, (0.445, 0.455), (0.450, 0.397), lw=0.50)

    kg_node(ax, 0.550, 0.800, 0.090, 0.078, "App", "robot / AR\nV2X", "#eef2f6", "app", fontsize=4.2)
    kg_node(ax, 0.680, 0.800, 0.090, 0.078, "QoS", "lat / rel.\njitter", "#f7f0f0", "qos", fontsize=4.2)
    kg_node(ax, 0.550, 0.665, 0.090, 0.078, "Slice", "URLLC\neMBB / mMTC", "#f1f5ed", "slice", fontsize=4.2)
    kg_node(ax, 0.680, 0.665, 0.090, 0.078, "Scenario", "factory\nedge / road", "#eef2f6", "scenario", fontsize=4.2)
    kg_node(ax, 0.550, 0.530, 0.090, 0.078, "Function", "UPF / sched.\nFW", "#f7f2e8", "function", fontsize=4.2)
    kg_node(ax, 0.680, 0.530, 0.090, 0.078, "Resource", "CPU / BW\nRAN", "#f2eff5", "resource", fontsize=4.2)
    rect(ax, 0.550, 0.360, 0.220, 0.075, "Relation set: requires, belongs_to, deployed_in\nsupported_by, consumes, conflicts_with", "#ffffff", fontsize=4.6)

    arrow(ax, (0.640, 0.839), (0.680, 0.839), "requires", (0.000, 0.023), lw=0.50, fontsize=4.0)
    arrow(ax, (0.595, 0.800), (0.595, 0.743), "belongs", (-0.035, 0.000), lw=0.50, fontsize=4.0)
    arrow(ax, (0.725, 0.800), (0.725, 0.743), "grounded", (0.041, 0.000), lw=0.50, fontsize=4.0)
    arrow(ax, (0.595, 0.665), (0.595, 0.608), "supported", (-0.040, 0.000), lw=0.50, fontsize=4.0)
    arrow(ax, (0.640, 0.569), (0.680, 0.569), "consumes", (0.000, 0.023), lw=0.50, fontsize=4.0)
    arrow(ax, (0.770, 0.832), (0.770, 0.569), "conflict", (0.030, 0.000), rad=-0.12, lw=0.50, fontsize=4.0)
    arrow(ax, (0.485, 0.485), (0.550, 0.839), lw=0.55, rad=-0.05)
    arrow(ax, (0.485, 0.485), (0.550, 0.704), lw=0.55, rad=-0.02)
    arrow(ax, (0.485, 0.485), (0.550, 0.569), lw=0.55, rad=0.03)

    tiny_icon_box(ax, 0.850, 0.820, "classify", "slice")
    tiny_icon_box(ax, 0.850, 0.740, "retrieve", "relation")
    tiny_icon_box(ax, 0.850, 0.660, "map", "function")
    tiny_icon_box(ax, 0.850, 0.580, "screen", "qos")
    tiny_icon_box(ax, 0.850, 0.500, "select", "resource")
    rect(ax, 0.840, 0.360, 0.120, 0.080, "Outputs\n$\\hat{y}_i$, $\\mathcal{K}_i$\nconflict / SLA", "#ffffff", fontsize=4.8)
    arrow(ax, (0.800, 0.840), (0.850, 0.840), lw=0.52)
    arrow(ax, (0.885, 0.820), (0.885, 0.780), lw=0.52)
    arrow(ax, (0.885, 0.740), (0.885, 0.700), lw=0.52)
    arrow(ax, (0.885, 0.660), (0.885, 0.620), lw=0.52)
    arrow(ax, (0.885, 0.580), (0.885, 0.540), lw=0.52)
    arrow(ax, (0.885, 0.500), (0.885, 0.440), lw=0.52)

    steps = [
        ("parse\nintent", 0.085),
        ("ground\nentities", 0.205),
        ("sample KG\nsubgraph", 0.330),
        ("encode by\nKCGT", 0.455),
        ("learn space\n$\\mathbf{z}_i,\\mathbf{r}_i$", 0.585),
        ("reuse\ntemplate", 0.710),
        ("screen\nconflict", 0.835),
        ("select\nSLA plan", 0.940),
    ]
    for label, x in steps:
        rect(ax, x - 0.047, 0.125, 0.094, 0.058, label, "#ffffff", fontsize=4.55)
    for (_, x_a), (_, x_b) in zip(steps[:-1], steps[1:]):
        arrow(ax, (x_a + 0.047, 0.154), (x_b - 0.047, 0.154), lw=0.50)
    ax.text(
        0.500,
        0.078,
        "feedback updates ontology aliases, service-template profiles, and conflict rules",
        ha="center",
        va="center",
        fontsize=4.3,
        color="#444444",
    )
    save(fig, "kg_schema")


def draw_architecture_legacy() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    group_box(ax, 0.015, 0.230, 0.145, 0.720, "A. Inputs", "#f4f6f8")
    group_box(ax, 0.185, 0.620, 0.340, 0.330, "B1. Semantic branch", "#f6f7f4")
    group_box(ax, 0.185, 0.350, 0.340, 0.240, "B2. Knowledge-graph branch", "#f6f7f4")
    group_box(ax, 0.185, 0.230, 0.340, 0.095, "B3. QoS branch", "#f6f7f4")
    group_box(ax, 0.548, 0.430, 0.185, 0.360, "C. Gated fusion", "#f4f6f8")
    group_box(ax, 0.755, 0.430, 0.225, 0.360, "D. Outputs", "#f6f4f1")
    group_box(ax, 0.185, 0.050, 0.795, 0.155, "E. Contrastive representation learning", "#f8f8f8", fontsize=6.2)

    photo_card(ax, 0.040, 0.830, 0.055, 0.055, "service", "service", fontsize=4.5)
    rect(ax, 0.035, 0.720, 0.100, 0.070, "$T_i$\nindustrial robot\nlow latency", "#ffffff", fontsize=5.0)
    rect(ax, 0.035, 0.545, 0.100, 0.075, "$\\mathcal{G}_i$\nApp-QoS-Slice\nFunction-Resource", "#ffffff", fontsize=4.7)
    rect(ax, 0.035, 0.375, 0.100, 0.065, "$\\mathbf{q}_i$\nlat, rel, bw\njitter, loss", "#ffffff", fontsize=4.8)
    rect(ax, 0.035, 0.270, 0.100, 0.045, "$y_i$, group id\nslice/profile", "#ffffff", fontsize=4.6)

    rect(ax, 0.205, 0.840, 0.060, 0.045, "tokenize\n$w_{i,l}$", "#ffffff", fontsize=4.6)
    rect(ax, 0.285, 0.840, 0.070, 0.045, "word emb.\n$\\mathbf{E}_w$", "#ffffff", fontsize=4.6)
    rect(ax, 0.375, 0.840, 0.060, 0.045, "mask\n$m_{i,l}$", "#ffffff", fontsize=4.6)
    rect(ax, 0.455, 0.840, 0.050, 0.045, "pool\n$\\bar{h}$", "#eef2f6", fontsize=4.6)
    rect(ax, 0.205, 0.745, 0.095, 0.050, "Transformer-ready\nor pretrained encoder", "#ffffff", fontsize=4.4)
    rect(ax, 0.330, 0.745, 0.095, 0.050, "masked mean\nsemantic state", "#ffffff", fontsize=4.4)
    rect(ax, 0.455, 0.745, 0.050, 0.050, "$\\mathbf{h}_i^t$", "#eef2f6", fontsize=5.5)

    rect(ax, 0.200, 0.500, 0.067, 0.045, "node/type\nembedding", "#ffffff", fontsize=4.3)
    rect(ax, 0.280, 0.500, 0.067, 0.045, "relation emb.\n$\\mathbf{e}_{uv}$", "#ffffff", fontsize=4.3)
    rect(ax, 0.360, 0.500, 0.078, 0.045, "relation-aware\n$H$-head attention", "#ffffff", fontsize=4.0)
    rect(ax, 0.451, 0.500, 0.057, 0.045, "pool\n$\\mathbf{h}_i^g$", "#eef2f6", fontsize=4.6)
    rect(ax, 0.205, 0.405, 0.075, 0.050, "$\\mathbf{q}_v=W_Qh_v$\nquery state", "#eef2f6", fontsize=4.0)
    rect(ax, 0.295, 0.405, 0.090, 0.050, "$\\mathbf{k}_{uv}=W_Kh_u+W_{Kr}e_{uv}$", "#eef2f6", fontsize=3.6)
    rect(ax, 0.400, 0.405, 0.105, 0.050, "$\\mathbf{v}_{uv}=W_Vh_u+W_{Vr}e_{uv}$\nresidual + FFN", "#ffffff", fontsize=3.55)

    rect(ax, 0.205, 0.235, 0.065, 0.040, "scale\n$\\tilde{q}$", "#ffffff", fontsize=4.6)
    rect(ax, 0.295, 0.235, 0.065, 0.040, "MLP\nBN/ReLU", "#ffffff", fontsize=4.6)
    rect(ax, 0.385, 0.235, 0.060, 0.040, "$\\mathbf{h}_i^q$", "#eef2f6", fontsize=5.2)

    rect(ax, 0.575, 0.680, 0.130, 0.045, "stack states\n$[\\mathbf{h}^t;\\mathbf{h}^g;\\mathbf{h}^q]$", "#ffffff", fontsize=4.5)
    rect(ax, 0.575, 0.615, 0.130, 0.045, "gate network\n$\\mathrm{softmax}(W_2\\sigma(W_1\\cdot))$", "#ffffff", fontsize=4.0)
    rect(ax, 0.575, 0.550, 0.130, 0.045, "modal weights\n$\\boldsymbol{\\alpha}_i=[\\alpha^t,\\alpha^g,\\alpha^q]$", "#ffffff", fontsize=4.0)
    rect(ax, 0.575, 0.485, 0.130, 0.045, "fused intent state\n$\\mathbf{z}_i\\in\\mathbb{R}^d$", "#eef2f6", fontsize=4.7)

    rect(ax, 0.782, 0.680, 0.078, 0.050, "classifier\n$C_\\phi$", "#f7f0f0", fontsize=4.7)
    rect(ax, 0.880, 0.680, 0.075, 0.050, "$\\hat{\\mathbf{y}}_i$\nintent class", "#ffffff", fontsize=4.5)
    rect(ax, 0.782, 0.585, 0.078, 0.050, "projection\n$P_\\psi$", "#f1f5ed", fontsize=4.7)
    rect(ax, 0.880, 0.585, 0.075, 0.050, "$\\mathbf{r}_i$\nunit vector", "#ffffff", fontsize=4.5)
    rect(ax, 0.782, 0.490, 0.078, 0.050, "Top-K\nindex", "#f1f5ed", fontsize=4.7)
    rect(ax, 0.880, 0.490, 0.075, 0.050, "templates\nconflicts\nSLA proxy", "#ffffff", fontsize=4.3)
    rect(ax, 0.800, 0.445, 0.135, 0.032, "outputs: $\\hat{y}_i,\\mathbf{r}_i,\\mathcal{K}_i$", "#ffffff", fontsize=4.4)

    rect(ax, 0.210, 0.110, 0.125, 0.050, "group/class positives\nsame reusable goal", "#ffffff", fontsize=4.1)
    rect(ax, 0.355, 0.110, 0.130, 0.050, "operational score\nKG + QoS + slice", "#ffffff", fontsize=4.1)
    rect(ax, 0.505, 0.110, 0.125, 0.050, "hard negatives\ndeclared + near", "#ffffff", fontsize=4.1)
    rect(ax, 0.650, 0.110, 0.135, 0.050, "OSCL\nweighted NCE + margin", "#ffffff", fontsize=4.1)
    rect(ax, 0.805, 0.110, 0.135, 0.050, "$\\mathcal{L}_{cls}+\\lambda(\\mathcal{L}_{oscl}+\\eta\\mathcal{L}_{mar})$", "#ffffff", fontsize=3.8)

    for y in (0.755, 0.582, 0.407):
        arrow(ax, (0.135, y), (0.205, y if y != 0.407 else 0.255), lw=0.62)
    arrow(ax, (0.265, 0.862), (0.285, 0.862), lw=0.55)
    arrow(ax, (0.355, 0.862), (0.375, 0.862), lw=0.55)
    arrow(ax, (0.435, 0.862), (0.455, 0.862), lw=0.55)
    arrow(ax, (0.300, 0.770), (0.330, 0.770), lw=0.55)
    arrow(ax, (0.425, 0.770), (0.455, 0.770), lw=0.55)
    arrow(ax, (0.267, 0.523), (0.280, 0.523), lw=0.55)
    arrow(ax, (0.347, 0.523), (0.360, 0.523), lw=0.55)
    arrow(ax, (0.438, 0.523), (0.451, 0.523), lw=0.55)
    arrow(ax, (0.270, 0.255), (0.295, 0.255), lw=0.55)
    arrow(ax, (0.360, 0.255), (0.385, 0.255), lw=0.55)
    arrow(ax, (0.505, 0.770), (0.575, 0.703), lw=0.60, rad=-0.07)
    arrow(ax, (0.505, 0.523), (0.575, 0.703), lw=0.60)
    arrow(ax, (0.445, 0.255), (0.575, 0.703), lw=0.60, rad=0.10)
    arrow(ax, (0.640, 0.680), (0.640, 0.660), lw=0.50)
    arrow(ax, (0.640, 0.615), (0.640, 0.595), lw=0.50)
    arrow(ax, (0.640, 0.550), (0.640, 0.530), lw=0.50)
    arrow(ax, (0.705, 0.508), (0.782, 0.705), lw=0.60, rad=-0.12)
    arrow(ax, (0.705, 0.508), (0.782, 0.610), lw=0.60, rad=-0.05)
    arrow(ax, (0.860, 0.705), (0.880, 0.705), lw=0.50)
    arrow(ax, (0.860, 0.610), (0.880, 0.610), lw=0.50)
    ax.plot([0.918, 0.918, 0.872], [0.585, 0.555, 0.555], color=INK, linewidth=0.50, zorder=1)
    arrow(ax, (0.872, 0.555), (0.860, 0.515), lw=0.50)
    arrow(ax, (0.860, 0.515), (0.880, 0.515), lw=0.50)
    arrow(ax, (0.335, 0.135), (0.355, 0.135), lw=0.50)
    arrow(ax, (0.485, 0.135), (0.505, 0.135), lw=0.50)
    arrow(ax, (0.630, 0.135), (0.650, 0.135), lw=0.50)
    arrow(ax, (0.785, 0.135), (0.805, 0.135), lw=0.50)

    save(fig, "kcgt_architecture")


def draw_architecture_case() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    group_box(ax, 0.015, 0.580, 0.255, 0.370, "A. Concrete intent input", "#f4f6f8", fontsize=6.2)
    group_box(ax, 0.295, 0.580, 0.245, 0.370, "B. Case grounding", "#f7f7f2", fontsize=6.2)
    group_box(ax, 0.565, 0.580, 0.200, 0.370, "C. KCGT encoding", "#f4f6f8", fontsize=6.2)
    group_box(ax, 0.790, 0.580, 0.195, 0.370, "D. Case outputs", "#f6f4f1", fontsize=6.2)
    group_box(ax, 0.015, 0.315, 0.970, 0.205, "E. Detailed processing trace", "#f8f8f8", fontsize=6.2)
    group_box(ax, 0.015, 0.055, 0.970, 0.205, "F. Representation learning signal for this case", "#f8f8f8", fontsize=6.2)

    photo_card(ax, 0.040, 0.815, 0.060, 0.060, "factory robot", "factory", fontsize=4.5)
    rect(ax, 0.105, 0.810, 0.150, 0.065, "Text intent\nindustrial robot needs\nURLLC control", "#ffffff", fontsize=4.4)
    rect(ax, 0.040, 0.705, 0.095, 0.060, "QoS vector\nlat=5 ms\nrel=99.999%", "#ffffff", fontsize=4.5)
    rect(ax, 0.150, 0.705, 0.105, 0.060, "Context\nfactory cell\nedge controller", "#ffffff", fontsize=4.4)
    rect(ax, 0.040, 0.625, 0.205, 0.050, "candidate label hint: industrial URLLC", "#eef2f6", fontsize=4.8)

    rect(ax, 0.320, 0.820, 0.085, 0.050, "entities\nApp=robot\nCtx=factory", "#ffffff", fontsize=4.1)
    rect(ax, 0.425, 0.820, 0.090, 0.050, "QoS concepts\nlow latency\nhigh reliability", "#ffffff", fontsize=4.2)
    rect(ax, 0.320, 0.720, 0.085, 0.050, "slice node\nURLLC", "#f1f5ed", fontsize=4.8)
    rect(ax, 0.425, 0.720, 0.090, 0.050, "function nodes\nUPF + scheduler", "#f7f2e8", fontsize=4.5)
    rect(ax, 0.320, 0.625, 0.195, 0.060, "sampled KG edges\nrobot->QoS; QoS->URLLC; URLLC->scheduler;\nfunction->edge CPU/BW", "#ffffff", fontsize=3.9)

    rect(ax, 0.590, 0.825, 0.145, 0.052, "semantic encoder\n$\\mathbf{h}^t$: control + reliability", "#ffffff", fontsize=4.3)
    rect(ax, 0.590, 0.745, 0.145, 0.052, "graph encoder\n$\\mathbf{h}^g$: URLLC-function-resource path", "#ffffff", fontsize=4.1)
    rect(ax, 0.590, 0.665, 0.145, 0.052, "QoS encoder\n$\\mathbf{h}^q$: tight latency/reliability", "#ffffff", fontsize=4.1)
    rect(ax, 0.590, 0.600, 0.145, 0.040, "gates: $\\alpha^t=.28,\\alpha^g=.39,\\alpha^q=.33$", "#eef2f6", fontsize=4.2)

    rect(ax, 0.815, 0.825, 0.140, 0.052, "classification\n$\\hat{y}$ = Industrial URLLC", "#f7f0f0", fontsize=4.5)
    rect(ax, 0.815, 0.745, 0.140, 0.052, "retrieval\nTop-1: factory URLLC template", "#f1f5ed", fontsize=4.4)
    rect(ax, 0.815, 0.665, 0.140, 0.052, "conflict screening\nQoS pass; slice pass\nfunction pass", "#ffffff", fontsize=3.9)
    rect(ax, 0.815, 0.600, 0.140, 0.040, "SLA-aware plan\nedge-UPF + scheduler", "#ffffff", fontsize=4.0)

    elbow_arrow(ax, [(0.255, 0.842), (0.285, 0.842), (0.285, 0.845), (0.320, 0.845)], lw=0.55)
    elbow_arrow(ax, [(0.135, 0.735), (0.285, 0.735), (0.285, 0.845), (0.425, 0.845)], lw=0.55)
    elbow_arrow(ax, [(0.245, 0.650), (0.285, 0.650), (0.285, 0.745), (0.320, 0.745)], lw=0.55)
    arrow(ax, (0.405, 0.745), (0.425, 0.745), lw=0.50)
    arrow(ax, (0.418, 0.720), (0.418, 0.685), lw=0.50)
    arrow(ax, (0.515, 0.655), (0.590, 0.850), lw=0.55, rad=-0.08)
    arrow(ax, (0.515, 0.655), (0.590, 0.770), lw=0.55)
    arrow(ax, (0.515, 0.655), (0.590, 0.690), lw=0.55, rad=0.08)
    arrow(ax, (0.735, 0.850), (0.815, 0.850), lw=0.55)
    arrow(ax, (0.735, 0.770), (0.815, 0.770), lw=0.55)
    arrow(ax, (0.735, 0.690), (0.815, 0.690), lw=0.55)

    trace = [
        ("1. parse", "robot,\nlow latency,\nreliable", 0.080),
        ("2. ground", "App:robot\nQoS:lat/rel\nScenario:factory", 0.215),
        ("3. build KG", "robot->QoS\nQoS->URLLC\nURLLC->scheduler", 0.355),
        ("4. encode", "$\\mathbf{h}^t,\\mathbf{h}^g,\\mathbf{h}^q$\n$\\in\\mathbb{R}^d$", 0.495),
        ("5. fuse", "$\\mathbf{z}_i=\\sum_m\\alpha^m\\mathbf{h}^m$\nKG/QoS weighted high", 0.635),
        ("6. retrieve", "$\\mathbf{r}_i=P(\\mathbf{z}_i)$\nTop-K templates", 0.775),
        ("7. decide", "template reuse\nconflict pass\nSLA candidate", 0.915),
    ]
    for title, body, x in trace:
        rect(ax, x - 0.055, 0.372, 0.110, 0.080, f"{title}\n{body}", "#ffffff", fontsize=4.1)
    for (_, _, x_a), (_, _, x_b) in zip(trace[:-1], trace[1:]):
        arrow(ax, (x_a + 0.055, 0.412), (x_b - 0.055, 0.412), lw=0.50)

    rect(ax, 0.045, 0.145, 0.160, 0.055, "positive examples\nrobot arm safety loop\nfactory motion control", "#ffffff", fontsize=4.4)
    rect(ax, 0.240, 0.145, 0.160, 0.055, "same service group\nindustrial control\nURLLC profile", "#ffffff", fontsize=4.4)
    rect(ax, 0.435, 0.145, 0.160, 0.055, "hard negatives\ncloud gaming: low latency\nbut eMBB objective", "#ffffff", fontsize=4.2)
    rect(ax, 0.630, 0.145, 0.160, 0.055, "contrastive effect\npull URLLC cases\npush eMBB neighbors", "#ffffff", fontsize=4.2)
    rect(ax, 0.825, 0.145, 0.125, 0.055, "training loss\n$\\mathcal{L}_{cls}+\\lambda\\mathcal{L}_{con}$", "#ffffff", fontsize=4.5)
    for x_a, x_b in [(0.205, 0.240), (0.400, 0.435), (0.595, 0.630), (0.790, 0.825)]:
        arrow(ax, (x_a, 0.172), (x_b, 0.172), lw=0.50)

    ax.text(
        0.500,
        0.087,
        "The same embedding supports classification, similarity retrieval, service-template reuse, conflict screening, and SLA-aware candidate selection.",
        ha="center",
        va="center",
        fontsize=4.6,
        color="#444444",
    )

    save(fig, "kcgt_architecture_case")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_std(value: str) -> tuple[float, float]:
    cleaned = value.replace("±", "+/-").replace("$\\pm$", "+/-")
    if "+/-" in cleaned:
        a, b = cleaned.split("+/-", 1)
        return float(a.strip()), float(b.strip())
    return float(cleaned.strip()), 0.0


def plot_retrieval_ablation() -> None:
    rows = read_rows(METHOD_NOVELTY / "method_novelty_summary.csv")
    display = {
        "full_structured_relgt": "KCGT",
        "plain_supcon_relgt": "w/o OSCL",
        "structured_rgcn": "R-GCN",
        "structured_relgt_shuffled": "shuffled KG",
    }
    rows = [r for r in rows if r["variant"] in display]
    labels = [display[r["variant"]] for r in rows]
    group = [(float(r["group_recall@1_mean"]), float(r["group_recall@1_std"])) for r in rows]
    hard = [(float(r["hard_precision@5_mean"]), float(r["hard_precision@5_std"])) for r in rows]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(3.55, 2.25))
    ax.bar(x - width / 2, [m for m, _ in group], width, yerr=[s for _, s in group], capsize=2, color=BLUE, edgecolor=INK, linewidth=0.45, hatch="//", label="Group R@1")
    ax.bar(x + width / 2, [m for m, _ in hard], width, yerr=[s for _, s in hard], capsize=2, color="#d9d9d9", edgecolor=INK, linewidth=0.45, hatch="..", label="Hard P@5")
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.2)
    save(fig, "final_retrieval_ablation")


def plot_robustness() -> None:
    rows = read_rows(ROBUSTNESS)
    label_map = {
        "text_paraphrase": "Paraphrase",
        "text_dropout": "Text dropout",
        "qos_noise_15pct": "QoS noise",
        "kg_edge_dropout_30pct": "KG edge",
        "kg_node_mask_30pct": "KG node",
        "relation_mask_has_constraint": "Constraint",
        "relation_mask_uses_capability": "Capability",
        "relation_mask_uses_resource_family": "Resource",
    }
    rows = [r for r in rows if r["mode"] != "clean"]
    labels = [label_map.get(r["mode"], r["mode"]) for r in rows]
    d_f1 = [float(r["delta_f1"]) for r in rows]
    d_r1 = [float(r["delta_recall@1"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(3.7, 2.35))
    ax.bar(x - width / 2, d_f1, width, color=GRAY, edgecolor=INK, linewidth=0.45, hatch="\\\\", label="$\\Delta$F1")
    ax.bar(x + width / 2, d_r1, width, color=ORANGE, edgecolor=INK, linewidth=0.45, hatch="..", label="$\\Delta$R@1")
    ax.axhline(0, color=INK, linewidth=0.75)
    ax.set_ylim(-0.56, 0.04)
    ax.set_ylabel("Change from clean")
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(pad=0.2)
    save(fig, "final_robustness_degradation")


def plot_tsne() -> None:
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except Exception:
        TSNE = None
        StandardScaler = None

    rows = read_rows(EMB)
    z_cols = [c for c in rows[0] if c.startswith("z")]
    x = np.array([[float(r[c]) for c in z_cols] for r in rows], dtype=float)
    labels = np.array([int(r["label"]) for r in rows])

    if StandardScaler is not None:
        x = StandardScaler().fit_transform(x)
    if TSNE is not None:
        coords = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=7).fit_transform(x)
    else:
        x = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        coords = x @ vt[:2].T

    palette = ["#4c78a8", "#9c755f", "#59a14f", "#b07aa1", "#8cd17d", "#79706e", "#f1ce63", "#76b7b2"]
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for lab in sorted(set(labels)):
        mask = labels == lab
        ax.scatter(coords[mask, 0], coords[mask, 1], s=7, color=palette[lab % len(palette)], alpha=0.72, linewidths=0, label=str(lab))
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(color="#d0d0d0", linewidth=0.35, alpha=0.45)
    ax.legend(
        title="Class",
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        fontsize=6.5,
        title_fontsize=6.8,
        handletextpad=0.35,
        columnspacing=0.75,
        labelspacing=0.35,
        borderaxespad=0.1,
    )
    fig.tight_layout(pad=0.2)
    save(fig, "final_tsne_neutral_kg")


def short_class(label: str) -> str:
    return (
        label.replace("AR/VR Immersive Service", "AR/VR")
        .replace("Autonomous Driving / V2X", "V2X")
        .replace("Cloud Gaming", "Gaming")
        .replace("Industrial URLLC", "Industrial")
        .replace("Massive IoT mMTC", "mMTC")
        .replace("Security-Critical Enterprise", "Security")
        .replace("Video Streaming eMBB", "Video")
    )


def plot_confusion() -> None:
    rows = read_rows(PRED)
    labels_full = sorted({r["true_class"] for r in rows})
    # Keep the original class-id order used by the previous figure/report.
    preferred = [
        "AR/VR Immersive Service",
        "Autonomous Driving / V2X",
        "Cloud Gaming",
        "Industrial URLLC",
        "Massive IoT mMTC",
        "Security-Critical Enterprise",
        "Telemedicine",
        "Video Streaming eMBB",
    ]
    labels_full = [x for x in preferred if x in labels_full] + [x for x in labels_full if x not in preferred]
    idx = {label: i for i, label in enumerate(labels_full)}
    cm = np.zeros((len(labels_full), len(labels_full)), dtype=int)
    for row in rows:
        cm[idx[row["true_class"]], idx[row["pred_class"]]] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)
    labels = [short_class(x) for x in labels_full]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.2, color="white" if cm_norm[i, j] > 0.55 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.2)
    save(fig, "final_confusion_kcgt")


def plot_gate_weights() -> None:
    rows = read_rows(GATE)
    labels = [short_class(r["true_class"]) for r in rows]
    text = np.array([float(r["text_gate"]) for r in rows])
    graph = np.array([float(r["graph_gate"]) for r in rows])
    qos = np.array([float(r["qos_gate"]) for r in rows])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3.75, 2.35))
    ax.bar(x, text, color="#d9d9d9", edgecolor=INK, linewidth=0.35, hatch="///", label="Text")
    ax.bar(x, graph, bottom=text, color=BLUE, edgecolor=INK, linewidth=0.35, hatch="..", label="Graph")
    ax.bar(x, qos, bottom=text + graph, color=GREEN, edgecolor=INK, linewidth=0.35, hatch="\\\\", label="QoS")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Gate weight")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.35, alpha=0.55)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(pad=0.2)
    save(fig, "final_gate_weights")


def plot_management_validation() -> None:
    rows = read_rows(MANAGEMENT)
    labels = ["Template@1", "Profile@1", "Fulfill@5", "Conflict F1", "Conflict AUROC"]
    keys = ["template_at_1", "profile_at_1", "fulfill_at_5", "conflict_f1", "conflict_auroc"]
    means = [float(next(r["mean"] for r in rows if r["metric"] == key)) for key in keys]
    stds = [float(next(r["std"] for r in rows if r["metric"] == key)) for key in keys]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(3.75, 2.35))
    ax.bar(x, means, yerr=stds, capsize=2, color=[BLUE, GRAY, GREEN, ORANGE, "#a58a9d"], edgecolor=INK, linewidth=0.45, hatch=["//", "..", "\\\\", "xx", ""])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.4, alpha=0.6)
    fig.tight_layout(pad=0.2)
    save(fig, "final_management_validation")


def vector_strip(ax, x, y, w, h, values, colors=None, label=None, fontsize=4.0):
    values = np.asarray(values, dtype=float)
    if colors is None:
        cmap = plt.get_cmap("Blues")
        norm = (values - values.min()) / (np.ptp(values) + 1e-9)
        colors = [cmap(0.28 + 0.55 * value) for value in norm]
    cell_w = w / len(values)
    for idx, color in enumerate(colors):
        ax.add_patch(
            Rectangle(
                (x + idx * cell_w, y),
                cell_w,
                h,
                linewidth=0.35,
                edgecolor="#ffffff",
                facecolor=color,
                zorder=2,
            )
        )
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=0.65, edgecolor=INK, zorder=3))
    if label:
        ax.text(x + w / 2, y - 0.008, label, ha="center", va="top", fontsize=fontsize)


def gate_bar(ax, x, y, w, value, label, color, fontsize=4.0):
    ax.add_patch(Rectangle((x, y), w, 0.018, linewidth=0.55, edgecolor=INK, facecolor="#ffffff"))
    ax.add_patch(Rectangle((x, y), w * value, 0.018, linewidth=0, facecolor=color))
    ax.text(x - 0.004, y + 0.009, label, ha="right", va="center", fontsize=fontsize)
    ax.text(x + w + 0.004, y + 0.009, f"{value:.3f}", ha="left", va="center", fontsize=fontsize)


def dashed_arrow(ax, start, end, label=None, off=(0, 0), rad=0.0, lw=0.7, fontsize=4.2):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        linestyle=(0, (4, 3)),
        color="#666666",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + off[0],
            (start[1] + end[1]) / 2 + off[1],
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#555555",
        )
    return arr


def knowledge_graph_icon(ax, x, y, w, h, fontsize=3.6):
    nodes = {
        "App": (x + 0.18 * w, y + 0.52 * h, "#bdd7ee"),
        "QoS": (x + 0.50 * w, y + 0.78 * h, "#fff2cc"),
        "Slice": (x + 0.52 * w, y + 0.28 * h, "#e2f0d9"),
        "NF": (x + 0.83 * w, y + 0.62 * h, "#e4dfec"),
        "Res": (x + 0.82 * w, y + 0.18 * h, "#fce4d6"),
    }
    for a, b in (("App", "QoS"), ("App", "Slice"), ("QoS", "NF"), ("Slice", "NF"), ("Slice", "Res")):
        xa, ya, _ = nodes[a]
        xb, yb, _ = nodes[b]
        ax.plot([xa, xb], [ya, yb], color="#6b7478", linewidth=0.7, zorder=2)
    radius = min(w, h) * 0.105
    for label, (xn, yn, color) in nodes.items():
        ax.add_patch(Circle((xn, yn), radius, facecolor=color, edgecolor=INK, linewidth=0.65, zorder=3))
        ax.text(xn, yn, label, ha="center", va="center", fontsize=fontsize, zorder=4)


def neural_network_icon(ax, x, y, w, h):
    layers = [3, 5, 4, 2]
    xs = np.linspace(x + 0.08 * w, x + 0.92 * w, len(layers))
    coords = []
    for li, count in enumerate(layers):
        ys = np.linspace(y + 0.16 * h, y + 0.84 * h, count)
        coords.append([(xs[li], yy) for yy in ys])
    for left, right in zip(coords[:-1], coords[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#a7b3b8", linewidth=0.28, zorder=1)
    colors = ["#bdd7ee", "#e4dfec", "#e2f0d9", "#f4b183"]
    radius = min(w, h) * 0.035
    for li, layer in enumerate(coords):
        for xn, yn in layer:
            ax.add_patch(Circle((xn, yn), radius, facecolor=colors[li], edgecolor=INK, linewidth=0.45, zorder=3))


def database_icon(ax, x, y, w, h, label, fc="#ddebf7", fontsize=4.0):
    from matplotlib.patches import Arc, Ellipse

    ax.add_patch(Rectangle((x, y + 0.12 * h), w, 0.70 * h, facecolor=fc, edgecolor=INK, linewidth=0.7))
    ax.add_patch(Ellipse((x + w / 2, y + 0.82 * h), w, 0.24 * h, facecolor=fc, edgecolor=INK, linewidth=0.7))
    ax.add_patch(Arc((x + w / 2, y + 0.47 * h), w, 0.22 * h, theta1=180, theta2=360, color=INK, linewidth=0.45))
    ax.add_patch(Arc((x + w / 2, y + 0.22 * h), w, 0.22 * h, theta1=180, theta2=360, color=INK, linewidth=0.45))
    ax.text(x + w / 2, y - 0.012, label, ha="center", va="top", fontsize=fontsize)


def draw_kg_schema() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    group_box(ax, 0.012, 0.525, 0.976, 0.445, "PHASE I. REPRESENTATION PREPARATION", "#f5f6f7", fontsize=6.7)
    group_box(ax, 0.012, 0.035, 0.976, 0.450, "PHASE II. ONLINE INTENT MANAGEMENT", "#f7f7f4", fontsize=6.7)

    for x in (0.245, 0.500, 0.705):
        ax.plot([x, x], [0.555, 0.925], color="#c3c3c3", linewidth=0.55, linestyle=(0, (3, 3)))
    for x in (0.205, 0.380, 0.725):
        ax.plot([x, x], [0.070, 0.440], color="#c3c3c3", linewidth=0.55, linestyle=(0, (3, 3)))

    ax.text(0.030, 0.910, "A. Multi-domain intent records", ha="left", va="center", fontsize=5.8)
    photo_card(ax, 0.035, 0.830, 0.052, 0.052, "factory", "factory", fontsize=4.0)
    photo_card(ax, 0.105, 0.830, 0.052, 0.052, "AR/city", "city", fontsize=4.0)
    photo_card(ax, 0.175, 0.830, 0.052, 0.052, "V2X", "vehicle", fontsize=4.0)
    rect(
        ax,
        0.030,
        0.700,
        0.200,
        0.080,
        "Intent $T_i$\nclosed-loop control: tight delay,\nlow failure exposure, continuity",
        "#ffffff",
        fontsize=4.35,
    )
    rect(
        ax,
        0.030,
        0.590,
        0.200,
        0.070,
        "QoS $\\mathbf{q}_i$\n12.05 ms | 42.89 Mb/s | rel. 0.99685\njitter 7.41 ms | loss $1.14\\times10^{-4}$",
        "#ffffff",
        fontsize=4.15,
    )
    rect(ax, 0.045, 0.540, 0.170, 0.032, "context + slice/function/resource clues", "#eef2f6", fontsize=4.0)

    ax.text(0.265, 0.910, "B. Normalization and KG grounding", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.270, 0.835, 0.090, 0.042, "entity + QoS\nnormalization", "#ffffff", fontsize=3.9)
    rect(ax, 0.385, 0.835, 0.090, 0.042, "ontology\nalignment", "#ffffff", fontsize=3.9)
    knowledge_graph_icon(ax, 0.275, 0.625, 0.195, 0.170, fontsize=3.3)
    ax.text(0.372, 0.610, "typed intent knowledge graph", ha="center", va="top", fontsize=3.9)
    rect(ax, 0.285, 0.550, 0.175, 0.040, "grounded instance $(T_i,\\mathbf{q}_i,\\mathcal{G}_i)$", "#eef2f6", fontsize=3.9)
    arrow(ax, (0.315, 0.835), (0.335, 0.790), lw=0.50)
    arrow(ax, (0.430, 0.835), (0.410, 0.790), lw=0.50)

    ax.text(0.520, 0.910, "C. KCGT representation service", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.525, 0.843, 0.050, 0.038, "$T_i$", "#eef2f6", fontsize=4.5)
    rect(ax, 0.525, 0.788, 0.050, 0.038, "$\\mathcal{G}_i$", "#f1f5ed", fontsize=4.5)
    rect(ax, 0.525, 0.733, 0.050, 0.038, "$\\mathbf{q}_i$", "#f7f2e8", fontsize=4.5)
    neural_network_icon(ax, 0.595, 0.735, 0.085, 0.105)
    ax.text(0.637, 0.700, "KCGT: Rel-GT + gates + OSCL", ha="center", va="top", fontsize=3.9)
    for yy in (0.862, 0.807, 0.752):
        arrow(ax, (0.575, yy), (0.600, 0.788), lw=0.45)
    vector_strip(ax, 0.555, 0.625, 0.115, 0.030, [0.1, 0.7, 0.35, 0.9, 0.55, 0.25, 0.8, 0.45], label="$\\mathbf{r}_i$: reusable intent embedding", fontsize=3.9)
    rect(ax, 0.545, 0.550, 0.135, 0.040, "classification + retrieval geometry", "#eef2f6", fontsize=4.0)
    arrow(ax, (0.637, 0.725), (0.612, 0.655), lw=0.50)

    ax.text(0.725, 0.910, "D. Management service database", ha="left", va="center", fontsize=5.8)
    database_icon(ax, 0.745, 0.700, 0.085, 0.130, "embedding index", "#ddebf7", fontsize=3.8)
    database_icon(ax, 0.865, 0.700, 0.085, 0.130, "template catalog", "#e2f0d9", fontsize=3.8)
    database_icon(ax, 0.785, 0.555, 0.125, 0.110, "QoS, conflict, resource records", "#fff2cc", fontsize=3.7)
    arrow(ax, (0.680, 0.640), (0.745, 0.760), lw=0.50, rad=-0.10)
    arrow(ax, (0.830, 0.760), (0.865, 0.760), lw=0.48)
    arrow(ax, (0.905, 0.700), (0.865, 0.655), lw=0.48)

    ax.text(0.030, 0.425, "E. New intent request", ha="left", va="center", fontsize=5.8)
    photo_card(ax, 0.035, 0.335, 0.055, 0.055, "factory", "factory", fontsize=3.9)
    rect(ax, 0.098, 0.315, 0.092, 0.090, "unseen wording\ntight delay control\n+ low failure exposure", "#ffffff", fontsize=3.9)
    rect(ax, 0.040, 0.235, 0.145, 0.050, "QoS + context\n12 ms, rel. 0.997, factory", "#ffffff", fontsize=3.9)
    rect(ax, 0.055, 0.145, 0.115, 0.050, "normalized query\n$(T_q,\\mathbf{q}_q,\\mathcal{G}_q)$", "#eef2f6", fontsize=4.0)
    arrow(ax, (0.112, 0.235), (0.112, 0.195), lw=0.50)

    ax.text(0.225, 0.425, "F. Query encoding", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.235, 0.300, 0.110, 0.075, "frozen KCGT\ntext + KG + QoS", "#ffffff", fontsize=4.4)
    vector_strip(ax, 0.235, 0.225, 0.110, 0.030, [0.12, 0.72, 0.33, 0.86, 0.51, 0.27, 0.78, 0.42], label="query $\\mathbf{r}_q$", fontsize=4.0)
    rect(ax, 0.245, 0.130, 0.090, 0.050, "predicted class\nIndustrial URLLC", "#eef2f6", fontsize=4.0)
    arrow(ax, (0.205, 0.170), (0.235, 0.337), lw=0.52, rad=-0.14)
    arrow(ax, (0.290, 0.300), (0.290, 0.255), lw=0.50)

    ax.text(0.400, 0.425, "G. Retrieval and feasibility decision", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.405, 0.315, 0.115, 0.070, "Top-$K$ shortlist\nURLLC retained\nV2X hard neighbor", "#ffffff", fontsize=3.9)
    rect(ax, 0.550, 0.315, 0.125, 0.070, "joint verification\nconflict + QoS/SLA\n+ capacity", "#fff2cc", fontsize=3.9)
    arrow(ax, (0.345, 0.240), (0.405, 0.350), lw=0.52, rad=-0.08)
    arrow(ax, (0.520, 0.350), (0.550, 0.350), lw=0.50)
    rect(ax, 0.430, 0.210, 0.225, 0.030, "FINAL MANAGEMENT OUTPUT", "#a9d18e", ec="#2f6335", lw=1.35, fontsize=4.25)
    rect(ax, 0.430, 0.140, 0.225, 0.070, "Decision: ADMIT | Service: Industrial URLLC\nDeployment: Edge UPF\nStatus: SLA feasible | conflict free", "#d9ead3", ec="#3f7d45", lw=1.35, fontsize=4.05)
    arrow(ax, (0.612, 0.315), (0.545, 0.240), lw=0.65)

    ax.text(0.745, 0.425, "H. Admission, deployment, and assurance", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.760, 0.340, 0.095, 0.050, "admission\naccept / defer", "#ffffff", fontsize=4.1)
    rect(ax, 0.875, 0.340, 0.090, 0.050, "slice\norchestrator", "#ffffff", fontsize=4.1)
    arrow(ax, (0.625, 0.150), (0.760, 0.365), lw=0.52, rad=-0.12)
    arrow(ax, (0.855, 0.365), (0.875, 0.365), lw=0.50)
    rect(ax, 0.745, 0.235, 0.060, 0.045, "RAN\nscheduler", "#eef2f6", fontsize=3.9)
    rect(ax, 0.815, 0.235, 0.060, 0.045, "Edge\nUPF", "#f1f5ed", fontsize=3.9)
    rect(ax, 0.885, 0.235, 0.075, 0.045, "compute +\nqueue", "#f7f2e8", fontsize=3.9)
    arrow(ax, (0.920, 0.340), (0.775, 0.280), lw=0.45, rad=0.10)
    arrow(ax, (0.920, 0.340), (0.845, 0.280), lw=0.45, rad=0.05)
    arrow(ax, (0.920, 0.340), (0.922, 0.280), lw=0.45)
    rect(ax, 0.790, 0.125, 0.135, 0.055, "KPI/SLA monitor\nlatency, loss, utilization", "#ffffff", fontsize=4.1)
    arrow(ax, (0.845, 0.235), (0.857, 0.180), lw=0.50)
    ax.plot([0.925, 0.988, 0.988], [0.150, 0.150, 0.736], color="#666666", linewidth=0.65, linestyle=(0, (4, 3)))
    dashed_arrow(ax, (0.988, 0.736), (0.968, 0.736), lw=0.65)
    ax.text(
        0.973,
        0.505,
        "telemetry / profile update",
        ha="right",
        va="center",
        fontsize=3.7,
        color="#555555",
    )
    save(fig, "kg_schema")


def draw_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    group_box(ax, 0.012, 0.520, 0.976, 0.450, "MAIN PATH. ONE INTENT FROM INPUT TO MANAGEMENT OUTPUT", "#f5f6f7", fontsize=6.6)
    group_box(ax, 0.012, 0.035, 0.390, 0.440, "E. Relation-aware Graph Transformer detail", "#f7f7f4", fontsize=5.9)
    group_box(ax, 0.420, 0.035, 0.350, 0.440, "F. Operationally structured contrastive learning", "#f7f7f4", fontsize=5.9)
    group_box(ax, 0.790, 0.035, 0.198, 0.440, "G. Train/inference contract", "#f7f7f4", fontsize=5.9)

    for x in (0.205, 0.525, 0.710):
        ax.plot([x, x], [0.550, 0.925], color="#c3c3c3", linewidth=0.55, linestyle=(0, (3, 3)))

    ax.text(0.030, 0.910, "A. Concrete intent instance", ha="left", va="center", fontsize=5.8)
    photo_card(ax, 0.035, 0.820, 0.052, 0.052, "factory", "factory", fontsize=3.9)
    rect(ax, 0.095, 0.790, 0.095, 0.095, "$T_i$\nclosed-loop control\ntight delay + continuity\nlow failure exposure", "#ffffff", fontsize=3.8)
    rect(
        ax,
        0.030,
        0.685,
        0.160,
        0.080,
        "$\\mathbf{q}_i$\nlatency 12.05 ms | bandwidth 42.89 Mb/s\nreliability 0.99685 | jitter 7.41 ms\nloss $1.14\\times10^{-4}$",
        "#ffffff",
        fontsize=3.75,
    )
    rect(
        ax,
        0.030,
        0.555,
        0.160,
        0.095,
        "$\\mathcal{G}_i$ (neutral KG)\ncontext: service request\ncapability: closed-loop control\nconstraints: latency, reliability\nresource: shared radio",
        "#ffffff",
        fontsize=3.35,
    )

    ax.text(0.225, 0.910, "B. Multi-modal encoders", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.225, 0.820, 0.075, 0.045, "tokens\nclosed | loop | delay", "#ffffff", fontsize=3.8)
    mini_transformer(ax, 0.322, 0.815, 0.096, 0.039, "semantic encoder:\nembedding + masked pool")
    vector_strip(ax, 0.445, 0.827, 0.060, 0.028, [0.2, 0.7, 0.4, 0.9, 0.3, 0.6], label="$\\mathbf{h}_i^t$", fontsize=3.8)
    arrow(ax, (0.300, 0.842), (0.320, 0.842), lw=0.48)
    arrow(ax, (0.420, 0.842), (0.445, 0.842), lw=0.48)

    rect(ax, 0.225, 0.710, 0.075, 0.055, "typed subgraph\ncontext - relation\n- capability/resource", "#ffffff", fontsize=3.55)
    mini_rel_gt(ax, 0.324, 0.707, 0.092, 0.050, "Rel-GT $\\times 2$:\nrelation-conditioned attention + FFN")
    vector_strip(ax, 0.445, 0.723, 0.060, 0.028, [0.4, 0.8, 0.25, 0.7, 0.9, 0.45], label="$\\mathbf{h}_i^g$", fontsize=3.8)
    arrow(ax, (0.300, 0.738), (0.320, 0.738), lw=0.48)
    arrow(ax, (0.420, 0.738), (0.445, 0.738), lw=0.48)

    rect(ax, 0.225, 0.605, 0.075, 0.048, "scaled QoS\n$\\widetilde{\\mathbf{q}}_i$", "#ffffff", fontsize=3.9)
    mini_mlp(ax, 0.323, 0.602, 0.094, 0.050, "2-layer MLP + BN/ReLU")
    vector_strip(ax, 0.445, 0.615, 0.060, 0.028, [0.3, 0.55, 0.85, 0.4, 0.65, 0.2], label="$\\mathbf{h}_i^q$", fontsize=3.8)
    arrow(ax, (0.300, 0.629), (0.320, 0.629), lw=0.48)
    arrow(ax, (0.420, 0.629), (0.445, 0.629), lw=0.48)

    ax.text(0.545, 0.910, "C. Sample-wise gated fusion", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.545, 0.815, 0.145, 0.045, "stack $[\\mathbf{h}_i^t;\\mathbf{h}_i^g;\\mathbf{h}_i^q]$\n$\\mathrm{softmax}(W_2\\sigma(W_1\\cdot))$", "#ffffff", fontsize=3.9)
    gate_bar(ax, 0.585, 0.765, 0.075, 0.245, "text", BLUE, fontsize=3.7)
    gate_bar(ax, 0.585, 0.728, 0.075, 0.479, "graph", GREEN, fontsize=3.7)
    gate_bar(ax, 0.585, 0.691, 0.075, 0.276, "QoS", ORANGE, fontsize=3.7)
    rect(ax, 0.555, 0.615, 0.125, 0.045, "fused state $\\mathbf{z}_i$\nprojection + $\\ell_2$ normalization", "#eef2f6", fontsize=3.9)
    vector_strip(ax, 0.570, 0.565, 0.095, 0.027, [0.15, 0.75, 0.35, 0.88, 0.52, 0.28, 0.79, 0.43], label="retrieval vector $\\mathbf{r}_i$", fontsize=3.8)
    arrow(ax, (0.505, 0.842), (0.545, 0.837), lw=0.45)
    arrow(ax, (0.505, 0.738), (0.545, 0.837), lw=0.45)
    arrow(ax, (0.505, 0.629), (0.545, 0.837), lw=0.45)
    arrow(ax, (0.617, 0.815), (0.617, 0.783), lw=0.45)
    arrow(ax, (0.617, 0.691), (0.617, 0.660), lw=0.45)
    arrow(ax, (0.617, 0.615), (0.617, 0.592), lw=0.45)

    ax.text(0.730, 0.910, "D. Concrete outputs", ha="left", va="center", fontsize=5.8)
    rect(ax, 0.735, 0.815, 0.105, 0.060, "classifier $C_\\phi$\nIndustrial URLLC\n$p=0.9967$", "#f7f0f0", fontsize=4.0)
    rect(ax, 0.860, 0.815, 0.105, 0.060, "Top-$K$ retrieval\nURLLC service\nprofiles", "#f1f5ed", fontsize=4.0)
    rect(ax, 0.735, 0.705, 0.105, 0.060, "conflict screen\nslice/function/QoS\ncompatibility", "#ffffff", fontsize=3.9)
    rect(ax, 0.860, 0.705, 0.105, 0.060, "SLA-aware rerank\nlatency/loss/resource\nverification", "#ffffff", fontsize=3.9)
    rect(ax, 0.760, 0.580, 0.180, 0.075, "management interface\n$\\hat{y}_i$, $\\mathbf{r}_i$, candidate set $\\mathcal{K}_i$\nfor template reuse and admission", "#eef2f6", fontsize=4.0)
    arrow(ax, (0.700, 0.845), (0.735, 0.845), lw=0.50)
    elbow_arrow(ax, [(0.665, 0.578), (0.700, 0.578), (0.700, 0.895), (0.912, 0.895), (0.912, 0.875)], lw=0.50)
    arrow(ax, (0.787, 0.815), (0.787, 0.765), lw=0.45)
    arrow(ax, (0.912, 0.815), (0.912, 0.765), lw=0.45)
    arrow(ax, (0.787, 0.705), (0.825, 0.655), lw=0.45)
    arrow(ax, (0.912, 0.705), (0.875, 0.655), lw=0.45)

    rect(ax, 0.035, 0.340, 0.075, 0.045, "service\nrequest", "#eef2f6", fontsize=4.0)
    rect(ax, 0.160, 0.370, 0.078, 0.045, "latency\nconstraint", "#f7f0f0", fontsize=4.0)
    rect(ax, 0.160, 0.290, 0.078, 0.045, "closed-loop\ncontrol", "#f1f5ed", fontsize=4.0)
    rect(ax, 0.285, 0.290, 0.075, 0.045, "shared\nradio", "#f2eff5", fontsize=4.0)
    arrow(ax, (0.110, 0.363), (0.160, 0.392), lw=0.45)
    arrow(ax, (0.110, 0.352), (0.160, 0.312), lw=0.45)
    arrow(ax, (0.238, 0.312), (0.285, 0.312), lw=0.45)
    ax.text(0.128, 0.414, "has constraint", ha="center", va="center", fontsize=3.2, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.text(0.130, 0.327, "uses capability", ha="center", va="center", fontsize=3.2, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.text(0.262, 0.345, "uses resource", ha="center", va="center", fontsize=3.2, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    rect(ax, 0.035, 0.190, 0.090, 0.055, "$\\mathbf{q}_v=W_Q\\mathbf{h}_v$\nnode query", "#ffffff", fontsize=3.8)
    rect(ax, 0.145, 0.190, 0.100, 0.055, "$\\mathbf{k}_{uv}=W_K\\mathbf{h}_u+W_{Kr}\\mathbf{e}_{uv}$\nrelation-conditioned key", "#ffffff", fontsize=3.35)
    rect(ax, 0.265, 0.190, 0.105, 0.055, "$\\mathbf{v}_{uv}=W_V\\mathbf{h}_u+W_{Vr}\\mathbf{e}_{uv}$\nrelation-conditioned value", "#ffffff", fontsize=3.35)
    rect(ax, 0.095, 0.090, 0.210, 0.055, "$\\alpha_{uv}=\\mathrm{softmax}(\\mathbf{q}_v^\\top\\mathbf{k}_{uv}/\\sqrt{d})$\n$H$ heads $\\rightarrow$ residual + LayerNorm + FFN $\\rightarrow \\mathbf{h}_i^g$", "#eef2f6", fontsize=3.65)
    arrow(ax, (0.080, 0.190), (0.160, 0.145), lw=0.45)
    arrow(ax, (0.195, 0.190), (0.195, 0.145), lw=0.45)
    arrow(ax, (0.317, 0.190), (0.240, 0.145), lw=0.45)

    rect(ax, 0.440, 0.335, 0.095, 0.075, "anchor $i$\nclosed-loop control\nlat. 12.05 ms\nIndustrial URLLC", "#eef2f6", fontsize=3.7)
    rect(ax, 0.555, 0.335, 0.095, 0.075, "positive $p$\nunseen composition\ntight delay + low failure\nsame group", "#f1f5ed", fontsize=3.7)
    rect(ax, 0.665, 0.335, 0.085, 0.075, "hard negative $n$\nroadside flow\nlat. 12.01 ms\nV2X", "#f7f0f0", fontsize=3.7)
    arrow(ax, (0.535, 0.372), (0.555, 0.372), lw=0.48)
    arrow(ax, (0.650, 0.372), (0.665, 0.372), lw=0.48)
    ax.text(0.545, 0.392, "pull", ha="center", va="center", fontsize=3.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.35))
    ax.text(0.6575, 0.392, "separate", ha="center", va="center", fontsize=3.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.35))
    rect(ax, 0.440, 0.220, 0.145, 0.075, "operational compatibility\nKG-relation overlap\n+ QoS distance\n+ slice/function/resource agreement", "#ffffff", fontsize=3.35)
    rect(ax, 0.605, 0.220, 0.145, 0.075, "hardness weighting\nnear QoS\n+ shared capabilities\nbut incompatible service goal", "#ffffff", fontsize=3.35)
    rect(ax, 0.455, 0.095, 0.270, 0.075, "OSCL objective\nweighted supervised NCE + hard-negative margin\n$\\mathcal{L}=\\mathcal{L}_{cls}+\\lambda(\\mathcal{L}_{oscl}+\\eta\\mathcal{L}_{mar})$", "#eef2f6", fontsize=3.65)
    arrow(ax, (0.520, 0.335), (0.520, 0.290), lw=0.45)
    arrow(ax, (0.700, 0.335), (0.675, 0.290), lw=0.45)
    arrow(ax, (0.520, 0.220), (0.555, 0.170), lw=0.45)
    arrow(ax, (0.675, 0.220), (0.635, 0.170), lw=0.45)

    rect(ax, 0.810, 0.340, 0.155, 0.060, "Training path\nlabels + groups + hard negatives\nclassification and OSCL losses", "#ffffff", fontsize=3.85)
    rect(ax, 0.810, 0.245, 0.155, 0.060, "Inference path\n$(T,\\mathbf{q},\\mathcal{G}) \\rightarrow \\hat{y},\\mathbf{r},\\mathcal{K}$\nno labels or positive pairs", "#ffffff", fontsize=3.85)
    rect(ax, 0.810, 0.135, 0.155, 0.060, "Actual trace shown above\nID: CHAL_HARD_C1_B007_P01\ncorrect class; graph gate dominates", "#eef2f6", fontsize=3.75)
    ax.plot([0.825, 0.855], [0.095, 0.095], color=INK, linewidth=0.8)
    ax.text(0.862, 0.095, "solid: forward/inference", ha="left", va="center", fontsize=3.8)
    ax.plot([0.825, 0.855], [0.065, 0.065], color="#666666", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(0.862, 0.065, "dashed: training/feedback", ha="left", va="center", fontsize=3.8)
    save(fig, "kcgt_architecture")


def main() -> None:
    setup()
    draw_kg_schema()
    draw_architecture()
    plot_tsne()
    plot_retrieval_ablation()
    plot_robustness()
    plot_confusion()
    plot_gate_weights()
    plot_management_validation()
    print(f"Rendered TNSM-style figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
