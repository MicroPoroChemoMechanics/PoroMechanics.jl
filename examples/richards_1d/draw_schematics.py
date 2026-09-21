"""Generate the Richards tutorial's conceptual SVG diagrams.

Run: python3 examples/richards_1d/draw_schematics.py
Uses only Python's standard library. These drawings are not simulation profiles.
"""
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / 'docs' / 'src' / 'assets'
INK, GRAY, BLUE, ORANGE, TEAL = '#193047', '#536879', '#216baf', '#c65a24', '#19796d'


class Diagram:
    def __init__(self, title, description, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" width="1000" height="{height}" role="img" aria-labelledby="title description">',
                      f'<title id="title">{escape(title)}</title><desc id="description">{escape(description)}</desc>', '<defs>']
        for name, color in [('blue', BLUE), ('orange', ORANGE), ('gray', GRAY)]:
            self.parts.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{color}"/></marker>')
        self.parts.append('</defs>')
        self.rect(1, 1, 998, height - 2, '#ffffff', '#ccd8e1')
        self.text(32, 42, title, 26, bold=True)

    def text(self, x, y, label, size=19, color=INK, anchor='start', bold=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{"bold" if bold else "normal"}">{escape(label)}</text>')

    def rect(self, x, y, w, h, fill, stroke='none'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}"/>')

    def poly(self, points, fill, stroke=GRAY):
        self.parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x, y in points)}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def line(self, x1, y1, x2, y2, color=GRAY, arrow=None, width=2):
        marker = f' marker-end="url(#{arrow})"' if arrow else ''
        self.parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{marker}/>')

    def save(self, name):
        (ASSETS / name).write_text('\n'.join(self.parts + ['</svg>']) + '\n')

def geometry():
    d = Diagram('Water enters a horizontal, unsaturated barrier',
                'A 20 cm barrier has a sealed left end and a saturated water supply at the right. Water travels toward smaller x. The lower bar shows initial bulk fractions of solid, liquid, and gas for porosity 0.30 and saturation 0.77752.', 665)
    d.text(32, 78, 'Fixed solid skeleton · fixed gas pressure · no gravity along the column', 19, GRAY)
    d.rect(80, 170, 700, 145, '#e4eeec', TEAL)
    for x in range(90, 645, 24):
        d.line(x, 184, x+18, 202, '#cadbd6', width=1)
        d.line(x, 282, x+18, 300, '#cadbd6', width=1)
    d.rect(650, 172, 128, 141, '#c1dff6')
    d.rect(800, 145, 160, 195, '#e3f2ff', BLUE)
    d.text(880, 192, 'Water', 23, BLUE, 'middle', True)
    d.text(880, 228, 'supply', 23, BLUE, 'middle', True)
    d.text(880, 271, 'pₗ = pɡ', 21, BLUE, 'middle')
    d.text(880, 304, 'Sₗ = 1', 21, BLUE, 'middle')
    d.text(345, 248, 'Porous barrier: L = 0.20 m', 24, TEAL, 'middle', True)
    d.line(80, 160, 80, 326, GRAY, width=5)
    for y in range(170, 325, 18):
        d.line(80, y, 65, y+12)
    d.text(80, 132, 'Sealed: Wₗ = 0', 20)
    d.line(760, 120, 480, 120, BLUE, 'blue', 3)
    d.text(617, 104, 'water flows toward smaller x', 18, BLUE, 'middle')
    d.line(80, 370, 780, 370, GRAY, 'gray')
    d.text(80, 398, 'x = 0', 18, anchor='middle')
    d.text(780, 398, 'x = L', 18, anchor='middle')
    d.text(32, 444, 'Initial interior: Sₗ ≈ 0.77752 is unsaturated, not water-free.', 22, bold=True)
    d.text(32, 475, 'Volumes in 1 m³ of material, with porosity φ = 0.30:', 19, GRAY)
    for x, w, fill in [(50, 630, '#dce3e9'), (680, 210, '#acd2f2'), (890, 60, '#fff0c2')]:
        d.rect(x, 500, w, 58, fill, '#ffffff')
    d.text(365, 536, 'Solid: 0.70 m³', 22, anchor='middle')
    d.text(785, 535, 'Liquid: 0.23326 m³', 17, anchor='middle')
    d.line(920, 560, 920, 584)
    d.text(950, 612, 'Gas: 0.06674 m³', 19, anchor='end')
    d.text(50, 608, 'Pores = liquid + gas = 0.30 m³', 21, BLUE)
    d.save('richards_geometry.svg')


def finite_volumes():
    d = Diagram('One pressure per node; a mass balance around each node',
                'An interior control volume extends halfway to neighboring nodes. Its stored water changes through fluxes at the left and right faces. Shared fluxes have opposite signs in adjacent balances. The model supplies storage and pressure-difference fluxes; VoronoiFVM supplies geometry and assembly.', 660)
    d.text(32, 79, 'Interior node: control-volume length ℓᵢ = h · endpoint nodes: ℓ = h/2', 20, GRAY)
    d.rect(370, 151, 260, 153, '#eaf3fb', BLUE)
    d.text(500, 187, 'Control volume i', 21, BLUE, 'middle', True)
    d.line(95, 249, 905, 249, GRAY, 'gray')
    for x, label in [(240, 'pᵢ₋₁'), (500, 'pᵢ'), (760, 'pᵢ₊₁')]:
        d.parts.append(f'<circle cx="{x}" cy="249" r="7" fill="{INK}"/>')
        d.text(x, 278, label, 23, anchor='middle')
    for x, label in [(370, 'x(i − 1/2)'), (630, 'x(i + 1/2)')]:
        d.line(x, 210, x, 311, BLUE, width=2)
        d.text(x, 336, label, 20, BLUE, 'middle')
    d.line(320, 218, 418, 218, ORANGE, 'orange', 3)
    d.line(580, 218, 678, 218, ORANGE, 'orange', 3)
    d.text(335, 139, 'W(i − 1/2)', 22, ORANGE, 'middle')
    d.text(665, 139, 'W(i + 1/2)', 22, ORANGE, 'middle')
    d.text(500, 377, 'Both fluxes are defined positive toward increasing x.', 20, GRAY, 'middle')
    d.rect(70, 409, 860, 93, '#fff3e5', ORANGE)
    d.text(500, 445, 'Accumulation + right outflow − left inflow = 0', 23, ORANGE, 'middle', True)
    d.text(500, 482, 'ℓᵢ [M(new) − M(old)] / Δt + W(right,new) − W(left,new) = 0', 22, anchor='middle')
    d.text(50, 548, 'Model callbacks', 21, TEAL, bold=True)
    d.text(50, 580, 'storage!: M = ρₗ φ Sₗ', 20)
    d.text(50, 612, 'flux!: K(p̄c) (pᵢ − pⱼ)', 20)
    d.line(375, 573, 475, 573, GRAY, 'gray', 3)
    d.text(515, 548, 'VoronoiFVM', 21, BLUE, bold=True)
    d.text(515, 580, 'Adds volume and edge geometry factors', 20)
    d.text(515, 612, 'Assembles and solves the nonlinear balances', 20)
    d.save('richards_finite_volumes.svg')


if __name__ == '__main__':
    geometry()
    finite_volumes()
