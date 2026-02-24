# Plane Strain J2 Plasticity (Q4 FEM, Global Newton)

**Author:** Habib Pouriayevali

A small-strain, plane-strain **J2 (von Mises) elastoplastic** finite element solver written in Python.  
The implementation includes **radial return mapping**, isotropic hardening, sparse global assembly, and a global Newton scheme.

---

## Overview

This code solves a 2D plane-strain tension problem using:

- **Q4 bilinear quadrilateral elements**
- **2×2 Gauss integration**
- **Small-strain J2 plasticity** with **isotropic hardening**
- **Backward-Euler radial return** (local update at Gauss points)
- **Sparse global stiffness assembly** and a global Newton iteration (with damping)

Outputs include:
- Traction vs. tip displacement curve
- von Mises stress field on a deformed mesh

---

## Governing Equations

### Kinematics (small strain, plane strain)
Strain vector used:
\[
\varepsilon_4 = [\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \gamma_{xy}]
\]
with plane strain condition \(\varepsilon_{zz}=0\) (computed implicitly through constitutive coupling).

### Linear elasticity (plane strain)
\[
\sigma = C : (\varepsilon - \varepsilon^p)
\]

### J2 Plasticity + Isotropic Hardening
Yield function:
\[
f = \sigma_{eq} - (\sigma_{y0} + H \alpha) \le 0
\]
with accumulated plastic strain \(\alpha\).

Equivalent von Mises stress:
\[
\sigma_{eq} = \sqrt{\frac{3}{2} \, s:s}
\]

Backward-Euler radial return update is used when \(f>0\).

---

## Numerical Implementation

### Elements and Integration
- Q4 bilinear elements
- 2×2 Gauss quadrature

### Local constitutive update
At each Gauss point:
- trial stress
- yield check
- radial return mapping
- update of deviatoric plastic strain and accumulated plastic strain

### Global solve (nonlinear)
- Assemble internal force \(F_{int}\) and tangent \(K\)
- Residual: \(R = F_{int} - F_{ext}\)
- Solve for increment: \(K \Delta u = -R\)
- Apply damping to improve robustness near yielding

---

## Problem Setup (Demo)

- Rectangular plate: \(L_x = 1.0\), \(L_y = 0.2\)
- Left edge fully fixed
- Uniform traction \(T_x\) applied on right edge
- Load stepping from 0 → \(T_{x,max}\)

Material:
- \(E = 210\,\text{GPa}\)
- \(\nu = 0.30\)
- \(\sigma_{y0} = 250\,\text{MPa}\)
- \(H = 1.0\,\text{GPa}\)

---

## Results

Example: von Mises stress plotted on the deformed mesh (scaled).

![von Mises on deformed mesh](von_mises_deformed.png)

---

## Dependencies

- numpy
- scipy
- matplotlib

Install:
```bash
pip install numpy scipy matplotlib
