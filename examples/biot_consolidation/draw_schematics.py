"""Generate conceptual Biot tutorial SVGs using only Python's standard library.

Run from the repository root: python3 examples/biot_consolidation/draw_schematics.py
Geometry follows the corner coordinates in ternay.msh; arrows are not computed data.
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
    d = Diagram('A saturated dam on a permeable foundation',
                'Cross-section of the supplied mesh. Reservoir water acts on the upstream concrete and rock. Downstream pressure is zero. Foundation sides prevent horizontal displacement; the base is fixed. The reservoir is outside the computational domain.', 665)
    d.text(32, 74, 'Mesh outline to scale · boundary arrows are schematic', 18, GRAY)
    # The first thirteen nodes are the concrete polygon's corners in mesh order.
    mesh = (Path(__file__).parent / 'ternay.msh').read_text().splitlines()
    start = mesh.index('$Nodes') + 2
    nodes = {int(row.split()[0]): tuple(map(float, row.split()[1:3]))
             for row in mesh[start:start + 17]}
    xy = lambda p: (170 + 6.3 * p[0], 112 + 6.3 * (517 - p[1]))
    concrete = [nodes[i] for i in range(1, 14)]
    water = [(-16, 517), nodes[6]] + [nodes[i] for i in range(5, 0, -1)] + [(-16, 476)]
    d.poly([xy(p) for p in water], '#e3f2ff', BLUE)
    d.poly([xy(p) for p in [(-16, 476), (43, 476), (43, 455), (-16, 455)]], '#e4eeec', TEAL)
    d.poly([xy(p) for p in concrete], '#f6ddc2', ORANGE)
    d.text(87, 154, 'Water', 19, BLUE)
    d.text(218, 282, 'Concrete', 19, ORANGE)
    d.text(250, 426, 'Rock', 21, TEAL, 'middle', True)
    for y, xedge in [(508, 4.9), (496, 4.9), (484, 2.4)]:
        x, yy = xy((xedge, y))
        d.line(x-55, yy, x-3, yy, BLUE, 'blue', 3)
    for xpos in [-11, -5]:
        x, y = xy((xpos, 476))
        d.line(x, y-45, x, y-4, BLUE, 'blue', 3)
    # Rollers at sides; hatching below the fixed base.
    for xpos in [-16, 43]:
        x, _ = xy((xpos, 476))
        for elev in [472, 465, 458]:
            _, y = xy((xpos, elev))
            offset = -8 if xpos < 0 else 8
            d.parts.append(f'<circle cx="{x+offset}" cy="{y}" r="5" fill="white" stroke="{GRAY}"/>')
    left, bottom = xy((-16, 455))
    right, _ = xy((43, 455))
    d.line(left, bottom+3, right, bottom+3, width=3)
    for x in range(int(left)+10, int(right), 16):
        d.line(x, bottom+3, x-10, bottom+14)
    d.text(250, 546, 'Fixed base: u₁ = u₂ = 0', 18, GRAY, 'middle')
    d.text(250, 575, 'Sides: u₁ = 0; base and sides: no flow', 17, GRAY, 'middle')
    d.line(*xy((-16, 517)), *xy((43, 517)), BLUE, width=1)
    d.text(447, 118, '517 m', 16, BLUE)
    d.text(447, 376, '476 m', 16, GRAY)
    d.text(447, 508, '455 m', 16, GRAY)
    d.rect(550, 106, 415, 153, '#eaf3fb')
    d.text(570, 138, 'UPSTREAM: reservoir loading', 20, BLUE, bold=True)
    d.text(570, 174, 'Pore pressure: p = ρg(H − y)', 20)
    d.text(570, 207, 'Surface traction: t = −p n', 20)
    d.text(570, 240, 'At y = 476 m: p = 0.41 MPa', 19)
    d.rect(550, 285, 415, 115, '#fff3e5')
    d.text(570, 320, 'DOWNSTREAM: drained surface', 20, ORANGE, bold=True)
    d.text(570, 355, 'p = 0 (reference pressure)', 20)
    d.text(570, 385, 'No applied surface traction', 19)
    d.text(550, 449, 'Water-filled pores in both materials.', 20, bold=True)
    d.text(550, 484, 'Zero pressure does not mean dry pores.', 19)
    d.text(550, 523, 'Shared interface: continuous u and p.', 19)
    d.text(32, 629, '2D plane strain · small deformation · no body gravity or self-weight in the equations', 19, GRAY)
    d.save('biot_geometry.svg')


def coupling():
    d = Diagram('Two fields, one coupled problem',
                'Solid deformation changes fluid storage. Pore pressure contributes to stress. Pressure gradients cause Darcy flow, which changes storage. Backward Euler solves both fields together.', 550)
    d.text(32, 77, 'A local undrained compression raises pressure; drainage then redistributes water.', 20, GRAY)
    d.rect(40, 133, 340, 216, '#fff3e5', ORANGE)
    d.rect(620, 133, 340, 216, '#eaf3fb', BLUE)
    d.text(210, 170, 'SOLID DEFORMATION', 22, ORANGE, 'middle', True)
    d.text(210, 212, 'Displacement u [m]', 22, anchor='middle')
    d.text(210, 252, 'Volume strain εᵥ = div u', 21, anchor='middle')
    d.text(210, 306, 'Force balance: div σ = 0', 20, anchor='middle')
    d.text(790, 170, 'PORE WATER', 22, BLUE, 'middle', True)
    d.text(790, 212, 'Pressure p [Pa]', 22, anchor='middle')
    d.text(790, 252, 'Fluid content ζ = b εᵥ + Np', 21, anchor='middle')
    d.text(790, 306, 'Darcy flow: q = −(k/μₗ) ∇p', 20, anchor='middle')
    d.line(390, 200, 610, 200, ORANGE, 'orange', 3)
    d.text(500, 185, 'changes storage', 18, ORANGE, 'middle')
    d.line(610, 281, 390, 281, BLUE, 'blue', 3)
    d.text(500, 312, 'changes stress', 18, BLUE, 'middle')
    d.rect(150, 390, 700, 112, '#e9f5f1', TEAL)
    d.text(500, 429, 'At each time step: solve for u and p together', 23, TEAL, 'middle', True)
    d.text(500, 470, '(K₁ + K₂ / Δt) Xⁿ⁺¹ = F + (K₂ / Δt) Xⁿ', 26, anchor='middle')
    d.line(210, 351, 210, 387, GRAY, 'gray')
    d.line(790, 351, 790, 387, GRAY, 'gray')
    d.save('biot_coupling.svg')


if __name__ == '__main__':
    geometry()
    coupling()
