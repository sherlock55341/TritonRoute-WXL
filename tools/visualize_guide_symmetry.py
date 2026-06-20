#!/usr/bin/env python3
import argparse
import html
import math
import re
from collections import Counter
from pathlib import Path


def natural_key(text):
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p for p in parts]


def parse_guide(path):
    nets = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        i += 1
        if not name or i >= len(lines) or lines[i].strip() != "(":
            continue
        i += 1
        rects = []
        while i < len(lines) and lines[i].strip() != ")":
            parts = lines[i].split()
            if len(parts) >= 5:
                try:
                    rects.append((int(parts[0]), int(parts[1]),
                                  int(parts[2]), int(parts[3]), parts[4]))
                except ValueError:
                    pass
            i += 1
        i += 1
        nets[name] = rects
    return nets


def parse_ap(path):
    aps = {}
    ap_path = Path(path)
    if not ap_path.exists():
        return aps
    for line in ap_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            aps.setdefault(parts[0], []).append((int(parts[1]), int(parts[2]), parts[3]))
        except ValueError:
            pass
    return aps


def default_ap_path(guide_path):
    guide = Path(guide_path)
    return guide.with_suffix(".ap")


def parse_axis(path):
    axes = {}
    axis_path = Path(path)
    if not axis_path.exists():
        return axes
    for line in axis_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            if parts[1] == "x":
                axes[parts[0]] = ("V", int(parts[2]) * 2)
            elif parts[1] == "y":
                axes[parts[0]] = ("H", int(parts[2]) * 2)
        except ValueError:
            pass
    return axes


def default_axis_path(guide_path):
    guide = Path(guide_path)
    return guide.with_suffix(".axis")


def mirror_rect(rect, orient, axis2):
    x1, y1, x2, y2, layer = rect
    if orient == "V":
        return (axis2 - x2, y1, axis2 - x1, y2, layer)
    return (x1, axis2 - y2, x2, axis2 - y1, layer)


def eval_axis(rects, orient, axis2):
    counts = Counter(rects)
    matched = 0
    for rect in sorted(list(counts)):
        n = counts.get(rect, 0)
        if n <= 0:
            continue
        mirrored = mirror_rect(rect, orient, axis2)
        if mirrored == rect:
            matched += n
            counts[rect] = 0
            continue
        k = min(n, counts.get(mirrored, 0))
        if k:
            counts[rect] -= k
            counts[mirrored] -= k
            matched += 2 * k
    return matched, len(rects) - matched


def axis_candidates(rects):
    out = []
    for x1, y1, x2, y2, _ in rects:
        out.append(("V", x1 + x2))
        out.append(("H", y1 + y2))
    for a in rects:
        for b in rects:
            out.append(("V", a[0] + b[2]))
            out.append(("V", a[2] + b[0]))
            out.append(("H", a[1] + b[3]))
            out.append(("H", a[3] + b[1]))
    return out


def best_axis(rects):
    best = None
    seen = set()
    for orient, axis2 in axis_candidates(rects):
        key = (orient, axis2)
        if key in seen:
            continue
        seen.add(key)
        matched, unmatched = eval_axis(rects, orient, axis2)
        score = (unmatched, -matched, abs(axis2))
        if best is None or score < best[0]:
            best = (score, orient, axis2, matched, unmatched)
    _, orient, axis2, matched, unmatched = best
    return orient, axis2, matched, unmatched


def project_rects_2d(rects):
    return sorted({(x1, y1, x2, y2) for x1, y1, x2, y2, _ in rects})


def axis_name(orient):
    return "y" if orient == "H" else "x"


def rect_bounds(rects, aps, orient, axis2, include_mirror):
    all_rects = list(rects)
    if include_mirror:
        all_rects += [mirror_rect(r, orient, axis2) for r in rects]
    min_x = min(r[0] for r in all_rects)
    min_y = min(r[1] for r in all_rects)
    max_x = max(r[2] for r in all_rects)
    max_y = max(r[3] for r in all_rects)
    for x, y, _ in aps:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    return min_x, min_y, max_x, max_y


def svg_rect(rect, bounds, scale, margin, klass, stroke, fill, opacity):
    min_x, min_y, _, max_y = bounds
    x1, y1, x2, y2 = rect[:4]
    x = margin + (x1 - min_x) * scale
    y = margin + (max_y - y2) * scale
    w = max(1.0, (x2 - x1) * scale)
    h = max(1.0, (y2 - y1) * scale)
    return (
        f'<rect class="{klass}" x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.4" opacity="{opacity}">'
        f'<title>({x1},{y1}) ({x2},{y2})</title></rect>'
    )


def svg_ap_marker(ap, bounds, scale, margin):
    min_x, _, _, max_y = bounds
    x, y, layer = ap
    sx = margin + (x - min_x) * scale
    sy = margin + (max_y - y) * scale
    size = 5.0
    return (
        f'<g class="ap">'
        f'<line x1="{sx - size:.2f}" y1="{sy - size:.2f}" '
        f'x2="{sx + size:.2f}" y2="{sy + size:.2f}"/>'
        f'<line x1="{sx - size:.2f}" y1="{sy + size:.2f}" '
        f'x2="{sx + size:.2f}" y2="{sy - size:.2f}"/>'
        f'<title>pref AP ({x},{y}) {html.escape(layer)}</title>'
        f'</g>'
    )


def write_svg(path, name, rects, aps, orient, axis2, matched, unmatched, include_mirror):
    margin = 28
    rects_2d = project_rects_2d(rects)
    mirror_rects_2d = []
    if include_mirror:
        mirror_rects_2d = project_rects_2d(
            [mirror_rect(r, orient, axis2) for r in rects])
    bounds_rects = [(x1, y1, x2, y2, "") for x1, y1, x2, y2 in rects_2d]
    if include_mirror:
        bounds_rects += [(x1, y1, x2, y2, "") for x1, y1, x2, y2 in mirror_rects_2d]
    bounds = rect_bounds(bounds_rects, aps, orient, axis2, False)
    min_x, min_y, max_x, max_y = bounds
    span_x = max(1, max_x - min_x)
    span_y = max(1, max_y - min_y)
    scale = min(900 / span_x, 620 / span_y)
    width = math.ceil(span_x * scale + margin * 2)
    height = math.ceil(span_y * scale + margin * 2 + 58)
    axis = axis2 / 2
    axis_label = axis_name(orient)

    body = []
    body.append(f'<text x="{margin}" y="20" class="title">{html.escape(name)}</text>')
    body.append(
        f'<text x="{margin}" y="42" class="meta">router axis={axis_label}={axis:g}, '
        f'2D guide boxes={len(rects_2d)}, source rects={len(rects)}, '
        f'pref APs={len(aps)}</text>'
    )

    if orient == "V":
        x = margin + (axis - min_x) * scale
        body.append(f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" '
                    f'y2="{height - margin}" class="axis"/>')
        body.append(f'<text x="{x + 8:.2f}" y="{height - margin - 8}" '
                    f'class="axis-label">axis x={axis:g}</text>')
    else:
        y = margin + (max_y - axis) * scale
        body.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width - margin}" '
                    f'y2="{y:.2f}" class="axis"/>')
        body.append(f'<text x="{margin + 8}" y="{y - 8:.2f}" '
                    f'class="axis-label">axis y={axis:g}</text>')

    if include_mirror:
        for rect in mirror_rects_2d:
            body.append(svg_rect(rect, bounds, scale, margin,
                                 "mirror", "#ef4444", "none", "0.75"))

    for rect in rects_2d:
        body.append(svg_rect(rect, bounds, scale, margin,
                             "guide", "#1d4ed8", "#60a5fa", "0.55"))

    for ap in aps:
        body.append(svg_ap_marker(ap, bounds, scale, margin))

    legend = []
    legend.append(f'<text x="{margin}" y="{height - 16}" class="legend">blue: 2D guide footprint; yellow x: pref AP; dashed black line: symmetry axis</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 16px sans-serif; fill: #0f172a; }}
  .meta, .legend, .axis-label {{ font: 12px sans-serif; fill: #334155; }}
  .axis {{ stroke: #111827; stroke-width: 2; stroke-dasharray: 8 6; }}
  .guide {{ vector-effect: non-scaling-stroke; }}
  .mirror {{ vector-effect: non-scaling-stroke; stroke-dasharray: 5 3; }}
  .ap line {{ stroke: #facc15; stroke-width: 3; stroke-linecap: round; vector-effect: non-scaling-stroke; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
{chr(10).join(body)}
{chr(10).join(legend)}
</svg>
'''
    path.write_text(svg)


def write_index(path, title, rows):
    items = []
    for row in rows:
        items.append(
            "<tr>"
            f"<td><a href='{html.escape(row['svg'])}'>{html.escape(row['name'])}</a></td>"
            f"<td>{html.escape(row['guide'])}</td>"
            f"<td>{html.escape(row['axis_name'])}={row['axis']:g}</td>"
            f"<td>{row['aps']}</td>"
            f"<td>{row['boxes_2d']}</td>"
            f"<td>{row['source_rects']}</td>"
            "</tr>"
        )
    index = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font: 14px system-ui, sans-serif; margin: 24px; color: #0f172a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 7px 8px; text-align: left; }}
    th {{ background: #f8fafc; position: sticky; top: 0; }}
    a {{ color: #2563eb; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Each SVG shows the 2D guide footprint, pref AP markers, and the router-reported symmetry axis.</p>
  <table>
    <thead><tr><th>Net</th><th>Guide</th><th>Axis</th><th>Pref APs</th><th>2D boxes</th><th>Source rects</th></tr></thead>
    <tbody>
      {chr(10).join(items)}
    </tbody>
  </table>
</body>
</html>
'''
    path.write_text(index)


def main():
    parser = argparse.ArgumentParser(description="Visualize Symmtry* guide symmetry as SVG/HTML.")
    parser.add_argument("guides", nargs="+", help="Guide files to visualize")
    parser.add_argument("-o", "--out-dir", default="build/guide_symmetry_check/vis",
                        help="Output directory")
    parser.add_argument("--with-mirror", action="store_true",
                        help="Also draw mirrored projection outlines")
    parser.add_argument("--ap", action="append", default=[],
                        help="AP sidecar file; defaults to each guide path with .ap suffix")
    parser.add_argument("--axis", action="append", default=[],
                        help="Axis sidecar file; defaults to each guide path with .axis suffix")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    explicit_aps = {}
    for ap_path in args.ap:
        explicit_aps.update(parse_ap(ap_path))
    explicit_axes = {}
    for axis_path in args.axis:
        explicit_axes.update(parse_axis(axis_path))
    for guide_path in args.guides:
        guide = Path(guide_path)
        nets = parse_guide(guide)
        aps_by_net = dict(explicit_aps)
        auto_ap_path = default_ap_path(guide)
        for net_name, aps in parse_ap(auto_ap_path).items():
            aps_by_net.setdefault(net_name, []).extend(aps)
        axes_by_net = dict(explicit_axes)
        axes_by_net.update(parse_axis(default_axis_path(guide)))
        for name in sorted((n for n in nets if n.startswith("Symmtry")), key=natural_key):
            rects = nets[name]
            if not rects:
                continue
            aps = aps_by_net.get(name, [])
            if name not in axes_by_net:
                raise SystemExit(f"Missing real axis for {name}; provide {default_axis_path(guide)} or --axis")
            orient, axis2 = axes_by_net[name]
            matched, unmatched = eval_axis(rects, orient, axis2)
            stem = f"{guide.stem}_{name}"
            svg_name = f"{stem}.svg"
            write_svg(out_dir / svg_name, name, rects, aps, orient, axis2,
                      matched, unmatched, args.with_mirror)
            rows.append({
                "name": name,
                "guide": str(guide),
                "svg": svg_name,
                "axis_name": axis_name(orient),
                "axis": axis2 / 2,
                "aps": len(aps),
                "boxes_2d": len(project_rects_2d(rects)),
                "source_rects": len(rects),
            })
    write_index(out_dir / "index.html", "Guide Symmetry Visualization", rows)
    print(f"Wrote {len(rows)} SVG files and {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
