"""Generate Darcy tutorial SVGs using the shared style of the Richards example.

Run: python3 examples/darcy_column/draw_schematics.py
Only Python's standard library is required; these are conceptual drawings.
"""
from pathlib import Path
import runpy

STYLE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'richards_1d' / 'draw_schematics.py'))
Diagram = STYLE['Diagram']
BLUE, GRAY, ORANGE, TEAL = (STYLE[c] for c in ('BLUE', 'GRAY', 'ORANGE', 'TEAL'))


def geometry():
    d = Diagram('A pressure ramp drives flow through a saturated column',
                'A one-meter column connects a zero-pressure bottom reservoir to a ramped top reservoir. The coordinate points upward and pressure-driven flow downward. Gravity is omitted; for a vertical interpretation pressure is excess over hydrostatic equilibrium.', 660)
    d.text(32, 80, 'One pressure unknown · constant storage · no explicit gravity term', 20, GRAY)
    d.rect(140, 130, 285, 65, '#e3f2ff', BLUE)
    d.text(282, 157, 'Top reservoir: region 2', 20, BLUE, 'middle', True)
    d.text(282, 182, 'p = p_top r(t)', 20, BLUE, 'middle')
    d.rect(177, 195, 210, 265, '#e4eeec', TEAL)
    d.text(282, 281, 'Saturated', 23, TEAL, 'middle', True)
    d.text(282, 317, 'porous column', 21, TEAL, 'middle')
    d.text(282, 354, 'L = 1 m', 22, TEAL, 'middle')
    d.rect(140, 460, 285, 65, '#e3f2ff', BLUE)
    d.text(282, 488, 'Bottom reservoir: region 1', 19, BLUE, 'middle', True)
    d.text(282, 514, 'p = 0', 21, BLUE, 'middle')
    d.line(100, 457, 100, 200, GRAY, 'gray', 3)
    d.text(88, 222, 'x = L', 18, GRAY, 'end')
    d.text(88, 451, 'x = 0', 18, GRAY, 'end')
    d.line(440, 237, 440, 428, BLUE, 'blue', 4)
    d.text(455, 331, 'q < 0', 22, BLUE)
    d.text(575, 139, 'Top pressure / p_top', 21, bold=True)
    d.line(590, 390, 945, 390, GRAY, 'gray')
    d.line(590, 390, 590, 185, GRAY, 'gray')
    d.line(590, 390, 760, 210, ORANGE, width=4)
    d.line(760, 210, 929, 210, ORANGE, width=4)
    d.text(575, 218, '1', 19, anchor='end')
    d.text(590, 419, '0', 19, anchor='middle')
    d.text(760, 419, '10 s = t_c', 19, anchor='middle')
    d.text(936, 419, 't', 20)
    d.text(585, 470, 'Initial pressure: p(x, 0) = 0', 21)
    d.text(585, 506, 'Both endpoints prescribe pressure.', 20, GRAY)
    d.rect(32, 555, 936, 77, '#fff3e5', ORANGE)
    d.text(500, 585, 'Vertical interpretation: p is excess above hydrostatic pressure.', 22, ORANGE, 'middle', True)
    d.text(500, 616, 'Steady state: linear pressure, constant downward flow; q = −10⁻⁴ m/s.', 21, anchor='middle')
    d.save('darcy_geometry.svg')


def finite_volumes():
    d = Diagram('Pressure storage and two shared interface fluxes',
                'The control volume around an interior pressure node extends halfway to its neighbors. Fluxes are oriented toward increasing x. Backward Euler balances the increment of stored fluid with the net new-time outflow.', 570)
    d.text(32, 81, '100 intervals · 101 pressure nodes · h = 0.01 m · unit cross-sectional area', 20, GRAY)
    d.rect(370, 152, 260, 147, '#eaf3fb', BLUE)
    d.text(500, 182, 'Stored increment: S pᵢ ℓᵢ', 20, BLUE, 'middle', True)
    d.line(100, 253, 900, 253, GRAY, 'gray')
    for x, label in [(240, 'pᵢ₋₁'), (500, 'pᵢ'), (760, 'pᵢ₊₁')]:
        d.parts.append(f'<circle cx="{x}" cy="253" r="7" fill="{BLUE}"/>')
        d.text(x, 285, label, 23, anchor='middle')
    for x in (370, 630):
        d.line(x, 194, x, 305, BLUE)
    d.line(323, 213, 415, 213, ORANGE, 'orange', 3)
    d.line(583, 213, 675, 213, ORANGE, 'orange', 3)
    d.text(312, 137, 'q(left)', 21, ORANGE, 'middle')
    d.text(693, 137, 'q(right)', 21, ORANGE, 'middle')
    d.text(500, 341, 'Both face fluxes use the positive-x orientation.', 21, GRAY, 'middle')
    d.rect(45, 368, 910, 77, '#fff3e5', ORANGE)
    d.text(500, 399, 'Stored-volume change + net outflow = 0', 24, ORANGE, 'middle', True)
    d.text(500, 430, 'ℓᵢ S [p(new) − p(old)] / Δt + q(right,new) − q(left,new) = 0', 21, anchor='middle')
    d.text(50, 490, 'Model: S p and (k/μ) (pᵢ − pⱼ)', 21, TEAL)
    d.text(50, 530, 'VoronoiFVM: volume weights + edge lengths + boundary constraints + solve', 21, BLUE)
    d.save('darcy_finite_volumes.svg')


if __name__ == '__main__':
    geometry()
    finite_volumes()
