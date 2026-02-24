Plane Strain J2 Plasticity (Q4 Finite Element Solver)
Author: Habib Pouriayevali

Overview
--------
This project implements a small-strain, plane-strain J2 (von Mises) elastoplastic finite element solver in Python.

The implementation includes:
- Q4 bilinear quadrilateral elements
- 2x2 Gauss integration
- J2 plasticity with isotropic hardening
- Backward-Euler radial return mapping
- Sparse global stiffness assembly (SciPy)
- Nonlinear global Newton solver with damping

The solver demonstrates a complete nonlinear FEM workflow with local constitutive updates at Gauss points and global equilibrium iterations.

Physical Model
--------------
Rectangular plate under uniaxial tension:
- Left edge fully fixed
- Uniform traction applied on the right edge
- Plane strain assumption
- Small strain kinematics

Example material parameters:
E      = 210 GPa
nu     = 0.30
sigy0  = 250 MPa
H      = 1.0 GPa

Load stepping is used to capture nonlinear elastoplastic behavior.

Governing Equations
-------------------
Elasticity (Plane Strain):
    sigma = C : (epsilon - epsilon_p)

Yield Function (J2 Plasticity):
    f = sigma_eq - (sigy0 + H * alpha)

Equivalent Stress:
    sigma_eq = sqrt(3/2 * s:s)

Backward-Euler radial return mapping is applied when yielding occurs.

Numerical Implementation
------------------------
- Structured rectangular mesh
- Local stress update at each Gauss point
- Internal force assembly
- Residual: R = Fint - Fext
- Solve K * du = -R using sparse linear solver
- Damped Newton updates

Note:
Under plane strain:
    epsilon_zz = 0
    sigma_zz is generally NOT zero.

The current implementation uses a modified Newton approach
(elastic tangent for robustness).

Outputs
-------
- Traction vs. tip displacement curve
- von Mises stress field on deformed mesh

Dependencies
------------
numpy
scipy
matplotlib

Install:
pip install numpy scipy matplotlib

Run:
python j2_plane_strain_q4.py
