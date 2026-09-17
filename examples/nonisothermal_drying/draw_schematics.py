"""Regenerate the tutorial's SVG diagrams with Python's standard library.

Run from any directory: python3 examples/nonisothermal_drying/draw_schematics.py
The diagrams are conceptual illustrations, not plots of simulation data.
"""
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / 'docs' / 'src' / 'assets'
INK, MUTED = '#193047', '#536879'
BLUE, ORANGE, TEAL = '#216baf', '#c65a24', '#19796d'


class Diagram:
    def __init__(self, title, description, height):
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" width="1000" height="{height}" role="img" aria-labelledby="title description">
<title id="title">{escape(title)}</title><desc id="description">{escape(description)}</desc>
<defs>''']
        for color, name in [(BLUE, 'blue'), (ORANGE, 'orange'), (TEAL, 'teal'), (MUTED, 'muted')]:
            self.parts.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{color}"/></marker>')
        self.parts.append('</defs>')
        self.rect(1, 1, 998, height-2, '#ffffff', '#ccd8e1', 16)
        self.text(35, 43, title, 26, weight='bold')

    def rect(self, x, y, w, h, fill, stroke='none', radius=0):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

    def text(self, x, y, value, size=20, color=INK, anchor='start', weight='normal'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')

    def line(self, x1, y1, x2, y2, color=MUTED, arrow=None, dash=None, width=2):
        extra = (f' marker-end="url(#{arrow})"' if arrow else '') + (f' stroke-dasharray="{dash}"' if dash else '')
        self.parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def circle(self, x, y, r, fill, stroke='none'):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def save(self, name):
        (ASSETS / name).write_text('\n'.join(self.parts + ['</svg>']) + '\n')


d = Diagram('One radial coordinate, a cylindrical physical domain',
            'Not to scale. A canister of radius 0.425 m is surrounded by clay to 1.225 m and rock to 10 m. Heat enters at the canister; the outer boundary is a reservoir. A volume-fraction bar distinguishes porosity from saturation.', 645)
d.text(35, 76, 'Cross-section perpendicular to the cylinder axis · schematic, not to scale', 18, MUTED)
for r, fill, stroke in [(172, '#e4eeec', TEAL), (108, '#f6ddc2', '#c19362'), (46, '#f6b190', ORANGE)]:
    d.circle(220, 278, r, fill, stroke)
d.text(220, 270, 'Hot', 21, anchor='middle', weight='bold')
d.text(220, 296, 'canister', 18, anchor='middle')
d.text(220, 197, 'CLAY', 19, anchor='middle', weight='bold')
d.text(220, 135, 'ROCK', 19, anchor='middle', weight='bold')
d.line(267, 280, 387, 280, ORANGE, 'orange', width=4)
d.text(310, 266, 'heat', 18, ORANGE, 'middle')
d.line(220, 355, 435, 355, MUTED, 'muted')
d.text(418, 341, 'r', 21, MUTED)
d.text(478, 128, 'The computational interval', 22, weight='bold')
d.rect(480, 180, 170, 86, '#f6ddc2', '#c19362')
d.rect(650, 180, 292, 86, '#e4eeec', TEAL)
d.text(565, 230, 'Clay', 23, anchor='middle')
d.text(795, 230, 'Rock', 23, anchor='middle')
for x, label in [(480, '0.425 m'), (650, '1.225 m'), (942, '10 m')]:
    d.line(x, 168, x, 278)
    d.text(x, 307, label, 19, anchor='middle')
d.line(445, 218, 480, 218, ORANGE, 'orange', width=4)
d.text(480, 350, 'Inner wall: heat in; water and air sealed', 19, ORANGE)
d.text(480, 382, 'Outer boundary: fixed pressures and T', 19, TEAL)
d.text(35, 483, 'Example: 1 m³ of material with porosity 0.30 and liquid saturation 0.80', 21, weight='bold')
for x, w, fill in [(50, 630, '#dce3e9'), (680, 216, '#acd2f2'), (896, 54, '#fff0c2')]:
    d.rect(x, 510, w, 55, fill, '#ffffff')
d.text(365, 545, 'Solid: 0.70 m³', 21, anchor='middle')
d.text(788, 545, 'Liquid: 0.24 m³', 19, anchor='middle')
d.line(923, 565, 923, 584)
d.text(945, 611, 'Gas: 0.06 m³', 19, anchor='end')
d.text(50, 610, 'Pores = liquid + gas = 0.30 m³', 21, BLUE)
d.save('drying_geometry.svg')

d = Diagram('How heat and moisture influence one another',
            'The three unknowns determine vapor pressure, capillary pressure, saturation, and dissolved air. These determine stored amounts and transport fluxes. The balances update the unknowns, closing the coupling loop.', 545)
d.text(35, 79, 'All arrows belong to one coupled solve at each time step.', 19, MUTED)
for x, y, w, h, fill, stroke in [(40, 145, 240, 230, '#eaf3fb', BLUE), (375, 125, 255, 270, '#fff3e5', ORANGE), (725, 145, 235, 230, '#e9f5f1', TEAL)]:
    d.rect(x, y, w, h, fill, stroke, 12)
d.text(160, 181, 'UNKNOWN STATE', 19, BLUE, 'middle', 'bold')
for y, label in [(228, 'Liquid pressure p_l'), (273, 'Air pressure p_a'), (318, 'Temperature T')]:
    d.text(160, y, label, 20, anchor='middle')
d.text(502, 161, 'LOCAL PHYSICS', 19, ORANGE, 'middle', 'bold')
for y, label in [(209, 'Vapor equilibrium'), (253, 'Capillary pressure'), (297, 'Liquid saturation'), (341, 'Dissolved air')]:
    d.text(502, y, label, 20, anchor='middle')
d.text(842, 181, 'STORAGE + FLUX', 19, TEAL, 'middle', 'bold')
for y, label in [(228, 'Water and air mass'), (273, 'Flow and diffusion'), (318, 'Entropy transport')]:
    d.text(842, y, label, 19, anchor='middle')
d.line(282, 260, 372, 260, BLUE, 'blue', width=3)
d.line(632, 260, 722, 260, ORANGE, 'orange', width=3)
d.parts.append(f'<path d="M842 377V451H160V378" fill="none" stroke="{TEAL}" stroke-width="3" marker-end="url(#teal)"/>')
d.rect(300, 428, 400, 43, '#ffffff')
d.text(500, 455, 'Balances update the unknown state', 21, TEAL, 'middle')
d.text(500, 510, 'Example: less liquid → lower thermal conductivity → a different temperature field', 19, MUTED, 'middle')
d.save('drying_coupling.svg')

d = Diagram('Finite volumes: one account for each radial shell',
            'Node i owns a shell between the two midpoint faces. The balance is accumulation plus right-face outflow minus left-face inflow. A shared face flux enters neighboring accounts with opposite signs. Storage, edge flux, and boundary conditions map to the three model callbacks.', 580)
d.text(35, 78, 'An interior shell in one material; each node carries p_l, p_a, and T.', 19, MUTED)
d.rect(310, 138, 260, 128, '#fff0db', '#d3913f', 4)
for x, label in [(310, 'r_(i−1/2)'), (570, 'r_(i+1/2)')]:
    d.line(x, 130, x, 318, '#af8048', dash='6 5')
    d.text(x, 116, label, 19, anchor='middle')
d.line(95, 209, 865, 209, MUTED, 'muted')
d.text(895, 216, 'r', 22)
for x, label in [(180, 'i−1'), (440, 'i'), (700, 'i+1')]:
    d.circle(x, 209, 8, BLUE)
    d.text(x, 185, f'Node {label}', 20, BLUE, 'middle', 'bold')
d.text(440, 250, 'Stored amount: V_i B(u_i)', 18, anchor='middle')
d.line(260, 294, 360, 294, TEAL, 'teal', width=4)
d.line(520, 294, 620, 294, ORANGE, 'orange', width=4)
d.text(218, 341, 'Left-face inflow', 19, TEAL, 'middle')
d.text(662, 341, 'Right-face outflow', 19, ORANGE, 'middle')
d.text(500, 387, 'Accumulation + right outflow − left inflow = 0', 24, anchor='middle', weight='bold')
for x, title, l1, l2, fill, color in [
    (35, 'storage!', 'Amount per bulk volume', 'Solver adds volume / Δt', '#fff3e5', ORANGE),
    (355, 'flux!', 'Coefficient × state difference', 'Solver adds area / distance', '#eaf3fb', BLUE),
    (675, 'bcondition!', 'Values or boundary fluxes', 'Only at the domain ends', '#e9f5f1', TEAL),
]:
    d.rect(x, 422, 290, 123, fill, color, 10)
    d.text(x+145, 455, title, 23, color, 'middle', 'bold')
    d.text(x+145, 492, l1, 18, anchor='middle')
    d.text(x+145, 522, l2, 18, anchor='middle')
d.save('drying_finite_volumes.svg')
