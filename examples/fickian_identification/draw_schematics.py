"""Generate the identification tutorial's conceptual SVGs.

Run: python3 examples/fickian_identification/draw_schematics.py
Uses the Richards example's drawing style and only Python's standard library.
"""
from pathlib import Path
import runpy

STYLE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'richards_1d' / 'draw_schematics.py'))
Diagram = STYLE['Diagram']
BLUE, GRAY, ORANGE, TEAL = (STYLE[c] for c in ('BLUE', 'GRAY', 'ORANGE', 'TEAL'))


def geometry():
    d = Diagram('Infer diffusivity from concentrations at several depths and ages',
                'A saturated one-meter specimen has a fixed-concentration inlet at x equals zero and a sealed far end. Six point measurements at depths from 0.02 to 0.30 meters are taken at two ages, producing twelve observations. The near-inlet region is enlarged below.', 625)
    d.text(32, 80, 'One solute · constant porosity · no advection, reaction, or sorption', 20, GRAY)
    d.rect(38, 127, 148, 150, '#e3f2ff', BLUE)
    d.text(112, 170, 'Reservoir', 22, BLUE, 'middle', True)
    d.text(112, 208, 'c = c_in', 23, BLUE, 'middle')
    d.text(112, 248, 'region 1', 19, BLUE, 'middle')
    d.rect(199, 148, 741, 108, '#e4eeec', TEAL)
    d.text(560, 194, 'Initially tracer-free: c(x, 0) = 0 inside', 23, TEAL, 'middle')
    d.text(560, 231, 'Stored solute per bulk volume = φ c', 21, TEAL, 'middle')
    d.line(940, 137, 940, 266, GRAY, width=5)
    d.text(936, 117, 'Sealed: j = 0', 20, anchor='end')
    d.line(206, 290, 938, 290, GRAY, 'gray')
    d.text(206, 319, 'x = 0', 19)
    d.text(938, 319, 'x = 1 m', 19, anchor='end')
    d.line(218, 120, 412, 120, BLUE, 'blue', 3)
    d.text(428, 125, 'solute enters: j > 0', 20, BLUE)
    d.text(38, 363, 'Sampling region enlarged: 0 to 0.30 m', 22, bold=True)
    d.line(120, 419, 922, 419)
    for depth in (.02, .05, .10, .15, .20, .30):
        x = 120 + 2600 * depth
        d.line(x, 401, x, 433, ORANGE, width=3)
        d.parts.append(f'<circle cx="{x}" cy="419" r="6" fill="{ORANGE}"/>')
        d.text(x, 463, f'{depth:.2f}', 19, anchor='middle')
    d.rect(38, 501, 924, 94, '#eaf3fb', BLUE)
    d.text(500, 535, '6 depths × 2 ages = 12 concentration observations', 24, BLUE, 'middle', True)
    d.text(500, 571, 'Ages: 2 × 10⁷ s and 10⁸ s · noise standard deviation: 0.005 mol/m³', 21, anchor='middle')
    d.save('identification_geometry.svg')


def workflow():
    d = Diagram('From a parameter guess to a calibrated diffusion model',
                'Scaled diffusivity and inlet concentration enter the finite volume and time-integration model. Twelve predictions are compared with observations. ForwardDiff propagates parameter derivatives through the solve. A damped least-squares step proposes parameters, accepted only for positive values and lower cost. The fitted Jacobian is also used to estimate local covariance.', 590)
    d.text(32, 81, 'ForwardDiff carries parameter sensitivities through the same forward calculation.', 20, GRAY)
    boxes = [(35, 'Trial parameters', 'θ₁ = D / D*', 'θ₂ = c_in / c*'),
             (360, 'Forward model', 'Finite volumes + time solve', 'Sample at depths and ages'),
             (685, 'Compare with data', 'r = prediction − observation', 'J = ∂ prediction / ∂θ')]
    for x, title, first, second in boxes:
        d.rect(x, 128, 280, 146, '#eaf3fb', BLUE)
        d.text(x + 140, 165, title, 23, BLUE, 'middle', True)
        d.text(x + 140, 209, first, 18, anchor='middle')
        d.text(x + 140, 243, second, 18, anchor='middle')
    d.line(316, 198, 356, 198, BLUE, 'blue', 3)
    d.line(641, 198, 681, 198, BLUE, 'blue', 3)
    d.line(825, 276, 825, 329, ORANGE, 'orange', 3)
    d.rect(360, 337, 605, 126, '#fff3e5', ORANGE)
    d.text(662, 370, 'Damped least-squares update', 23, ORANGE, 'middle', True)
    d.text(662, 406, '[JᵀJ + λ diag(JᵀJ)] δ = −Jᵀr', 23, anchor='middle')
    d.text(662, 441, 'Accept positive parameters only when cost decreases.', 20, anchor='middle')
    d.line(360, 402, 176, 402, ORANGE, width=3)
    d.line(176, 402, 176, 279, ORANGE, 'orange', 3)
    d.text(176, 438, 'Next trial', 21, ORANGE, 'middle')
    d.rect(35, 498, 930, 62, '#e4eeec', TEAL)
    d.text(500, 537, 'At the fit: inspect residuals, rank of J, local uncertainty, and mesh bias.', 22, TEAL, 'middle')
    d.save('identification_workflow.svg')


if __name__ == '__main__':
    geometry()
    workflow()
