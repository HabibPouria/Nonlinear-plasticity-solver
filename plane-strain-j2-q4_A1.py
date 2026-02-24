import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# ============================================================
# Q4 kinematics
# Node order: 0:(-1,-1), 1:(+1,-1), 2:(+1,+1), 3:(-1,+1)
# ============================================================
def dN_dxi_eta_Q4(xi, eta):
    dN_dxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)], dtype=float)
    dN_deta= 0.25 * np.array([-(1-xi), -(1+xi), (1+xi),  (1-xi)], dtype=float)
    return dN_dxi, dN_deta

def B_detJ_Q4_plane_strain(xy, xi, eta):
    """
    Plane strain strain vector we use:
      eps4 = [exx, eyy, ezz, gxy]  where ezz=0
    B4 maps element DOFs u_e(8,) -> eps4(4,)
    """
    dN_dxi, dN_deta = dN_dxi_eta_Q4(xi, eta)
    x = xy[:,0]; y = xy[:,1]

    J = np.array([[np.dot(dN_dxi,x),  np.dot(dN_deta,x)],
                  [np.dot(dN_dxi,y),  np.dot(dN_deta,y)]], dtype=float)
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError(f"Non-positive detJ={detJ}. Check node ordering/distortion.")
    invJ = np.linalg.inv(J)

    grads = np.vstack((dN_dxi, dN_deta))  # (2,4)
    dN_dxdy = invJ @ grads                 # (2,4)
    dN_dx = dN_dxdy[0,:]
    dN_dy = dN_dxdy[1,:]

    B4 = np.zeros((4,8), dtype=float)
    for a in range(4):
        # exx
        B4[0,2*a]   = dN_dx[a]
        # eyy
        B4[1,2*a+1] = dN_dy[a]
        # ezz = 0 in plane strain -> row stays zero
        # gxy (engineering shear)
        B4[3,2*a]   = dN_dy[a]
        B4[3,2*a+1] = dN_dx[a]

    return B4, detJ

# ============================================================
# Elasticity: plane strain (4x4) for eps4=[exx, eyy, ezz, gxy]
# ============================================================
def C_plane_strain_4(E, nu):
    lam = E*nu/((1+nu)*(1-2*nu))
    mu  = E/(2*(1+nu))
    C = np.zeros((4,4), dtype=float)

    # normal part (xx,yy,zz)
    C[0,0] = lam + 2*mu; C[0,1] = lam;        C[0,2] = lam
    C[1,0] = lam;        C[1,1] = lam + 2*mu; C[1,2] = lam
    C[2,0] = lam;        C[2,1] = lam;        C[2,2] = lam + 2*mu

    # shear part (engineering shear gamma_xy)
    C[3,3] = mu
    return C, lam, mu

# ============================================================
# J2 plasticity (small strain) radial return in 3D
# We store epsp_dev as deviatoric plastic strain components:
#   ep_dev = [ep_xx, ep_yy, ep_zz, ep_xy]  (with trace=0)
# and alpha = accumulated eq plastic strain
#
# Strain input eps4 = [exx, eyy, ezz, gxy], where ezz=0
# Stress output sig4 = [sxx, syy, szz, txy]
# ============================================================
def deviatoric_from_sig4(sig4):
    sxx, syy, szz, txy = sig4
    m = (sxx + syy + szz) / 3.0
    return np.array([sxx-m, syy-m, szz-m, txy], dtype=float)

def j2_eq_from_dev4(sdev4):
    # J2 with engineering shear: s:s = sxx^2+syy^2+szz^2 + 2*txy^2
    sxx, syy, szz, txy = sdev4
    ss = sxx*sxx + syy*syy + szz*szz + 2.0*txy*txy
    return np.sqrt(1.5 * ss)

def return_map_J2_plane_strain(eps4, C4, mu, sigy0, H, epdev_old, alpha_old):
    """
    Backward-Euler radial return, isotropic hardening:
      f = sigma_eq - (sigy0 + H*alpha) <= 0
    Returns:
      sig4, epdev_new, alpha_new, Cep4 (algorithmic tangent 4x4)
    """
    # elastic trial: sigma_trial = C : (eps - epsp_dev)   (vol part handled by C)
    sig_trial = C4 @ (eps4 - epdev_old)
    s_trial = deviatoric_from_sig4(sig_trial)
    seq_trial = j2_eq_from_dev4(s_trial)
    f_trial = seq_trial - (sigy0 + H*alpha_old)

    if f_trial <= 0.0:
        # elastic
        return sig_trial, epdev_old, alpha_old, C4

    # plastic
    # For J2 radial return: delta_gamma = f_trial / (3*mu + H)
    dgamma = f_trial / (3.0*mu + H)

    # flow direction n = s_trial / ||s_trial||  in deviatoric space
    # handle safe division
    norm_s = np.sqrt(s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 + 2.0*s_trial[3]**2)
    if norm_s == 0.0:
        # extremely unlikely
        n = np.array([1.0, -1.0, 0.0, 0.0], dtype=float)
        # enforce deviatoric
        n[:3] -= np.mean(n[:3])
    else:
        # Note: use the same metric as norm_s above (shear weighted by 2)
        n = s_trial / norm_s

    # stress update: s_new = s_trial - 2*mu*dgamma*(3/2)*n? (careful with definitions)
    # With our n based on norm_s (not seq), use the standard radial return in deviatoric stress:
    # s_new = s_trial * (1 - 3*mu*dgamma/seq_trial)
    s_new = s_trial * (1.0 - (3.0*mu*dgamma)/seq_trial)

    # reconstruct full stress: keep mean stress from trial (plastic is deviatoric)
    m_trial = (sig_trial[0] + sig_trial[1] + sig_trial[2]) / 3.0
    sig_new = np.array([s_new[0]+m_trial, s_new[1]+m_trial, s_new[2]+m_trial, s_new[3]], dtype=float)

    # plastic strain deviatoric increment:
    # dep_dev = (3/2)*dgamma*(s_trial/seq_trial) in tensor form (engineering shear handled consistently)
    # Use n_seq = s_trial/seq_trial in deviatoric 2nd order sense:
    dep_dev = (1.5*dgamma/seq_trial) * s_trial
    epdev_new = epdev_old + dep_dev

    # enforce deviatoric trace exactly (numerical hygiene)
    tr = epdev_new[0] + epdev_new[1] + epdev_new[2]
    epdev_new[0] -= tr/3.0
    epdev_new[1] -= tr/3.0
    epdev_new[2] -= tr/3.0

    alpha_new = alpha_old + dgamma

    # Algorithmic tangent (simple, robust form):
    # Cep = C - (C : Pdev : C) / (H + 3*mu)
    # For this 4-component plane-strain representation, we approximate with:
    # Cep = C - (2*mu)^2/(H+3*mu) * (Pdev)
    # where Pdev projects onto deviatoric subspace (with shear weighting).
    #
    # This tangent is a good practical choice for stable global Newton.
    Pdev = np.zeros((4,4), dtype=float)
    # Deviatoric projector on normal components (xx,yy,zz)
    Pdev[0,0] = 2/3; Pdev[0,1] = -1/3; Pdev[0,2] = -1/3
    Pdev[1,0] = -1/3; Pdev[1,1] = 2/3; Pdev[1,2] = -1/3
    Pdev[2,0] = -1/3; Pdev[2,1] = -1/3; Pdev[2,2] = 2/3
    # Shear part is already deviatoric
    Pdev[3,3] = 1.0

# Robust choice for global Newton: use elastic tangent (modified Newton)
# This avoids inconsistent tangents causing divergence.
    return sig_new, epdev_new, alpha_new, C4

# ============================================================
# Mesh
# ============================================================
def make_rect_mesh_Q4(Lx, Ly, nx, ny):
    xs = np.linspace(0.0, Lx, nx+1)
    ys = np.linspace(0.0, Ly, ny+1)
    nodes = np.array([[x,y] for y in ys for x in xs], dtype=float)

    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j*(nx+1)+i
            n1 = n0+1
            n3 = (j+1)*(nx+1)+i
            n2 = n3+1
            elems.append([n0,n1,n2,n3])
    return nodes, np.array(elems, dtype=int)

def dofs_of_element(conn):
    dofs = []
    for n in conn:
        dofs.extend([2*n, 2*n+1])
    return np.array(dofs, dtype=int)

# ============================================================
# Nonlinear FEM solver (global Newton) for J2 plasticity
# ============================================================
class Q4PlaneStrainJ2Solver:
    def __init__(self, nodes, elems, E, nu, t, sigy0, H):
        self.nodes = nodes
        self.elems = elems
        self.E = E
        self.nu = nu
        self.t = t
        self.sigy0 = sigy0
        self.H = H

        self.C4, self.lam, self.mu = C_plane_strain_4(E, nu)

        self.n_nodes = nodes.shape[0]
        self.n_dofs = 2*self.n_nodes

        # 2x2 Gauss points
        gp = 1.0/np.sqrt(3.0)
        self.gauss = [(-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)]
        self.ngp = 4

        # Internal variables per element per gp
        ne = len(elems)
        self.epdev = np.zeros((ne, self.ngp, 4), dtype=float)  # deviatoric plastic strain (4 comps)
        self.alpha = np.zeros((ne, self.ngp), dtype=float)     # accumulated eq plastic strain

        self.F_ext = np.zeros(self.n_dofs, dtype=float)
        self.fixed = np.array([], dtype=int)

    def set_traction_right_edge(self, Lx, Ly, nx, ny, Tx):
        """Uniform traction Tx on right edge, consistent nodal forces on boundary segments."""
        self.F_ext[:] = 0.0
        right_nodes = np.array([j*(nx+1)+nx for j in range(ny+1)], dtype=int)
        dy = Ly/ny
        for j in range(ny):
            nA = right_nodes[j]
            nB = right_nodes[j+1]
            fseg = Tx * self.t * dy
            self.F_ext[2*nA] += 0.5*fseg
            self.F_ext[2*nB] += 0.5*fseg

    def fix_left_edge(self, nx, ny):
        left_nodes = np.array([j*(nx+1)+0 for j in range(ny+1)], dtype=int)
        fixed = []
        for n in left_nodes:
            fixed.extend([2*n, 2*n+1])
        self.fixed = np.array(sorted(set(fixed)), dtype=int)

    def assemble(self, u, ep_trial, alpha_trial):
        K = lil_matrix((self.n_dofs, self.n_dofs), dtype=float)
        Fint = np.zeros(self.n_dofs, dtype=float)

        ne = len(self.elems)
        ep_new = np.zeros_like(ep_trial)
        alpha_new = np.zeros_like(alpha_trial)

        # for postprocessing (optional)
        vm_elem = np.zeros(ne, dtype=float)

        for e_id, conn in enumerate(self.elems):
            xy = self.nodes[conn]
            edofs = dofs_of_element(conn)
            u_e = u[edofs]

            Ke = np.zeros((8,8), dtype=float)
            fe = np.zeros(8, dtype=float)
            vm_gp_list = []

            for gp_id, (xi,eta) in enumerate(self.gauss):
                B4, detJ = B_detJ_Q4_plane_strain(xy, xi, eta)
                eps4 = B4 @ u_e  # [exx, eyy, ezz(=0), gxy]

                sig4, epdev_new, alpha_g_new, Cep4 = return_map_J2_plane_strain(
                    eps4, self.C4, self.mu, self.sigy0, self.H,
                    ep_trial[e_id, gp_id, :], alpha_trial[e_id, gp_id]
                )

                ep_new[e_id, gp_id, :] = epdev_new
                alpha_new[e_id, gp_id] = alpha_g_new

                # Internal force: ∫ B^T * sigma dΩ
                # Here sigma4 = [sxx, syy, szz, txy], but B4 uses exx, eyy, ezz, gxy
                fe += (B4.T @ sig4) * self.t * detJ

                # Tangent: ∫ B^T * Cep * B dΩ
                Ke += (B4.T @ Cep4 @ B4) * self.t * detJ

                # von Mises (3D) from deviatoric stress
                sdev = deviatoric_from_sig4(sig4)
                seq = j2_eq_from_dev4(sdev)
                vm_gp_list.append(seq)

            # element average vm
            vm_elem[e_id] = np.mean(vm_gp_list)

            # assemble into global
            for a in range(8):
                ia = edofs[a]
                Fint[ia] += fe[a]
                for b in range(8):
                    ib = edofs[b]
                    K[ia, ib] += Ke[a, b]

        return Fint, K.tocsr(), ep_new, alpha_new, vm_elem

    def solve_loadstep(self, u0, max_iter=30, tol=1e-8):
        u = u0.copy()

        free = np.setdiff1d(np.arange(self.n_dofs), self.fixed)

        # trial state
        ep_trial = self.epdev.copy()
        alpha_trial = self.alpha.copy()

        for it in range(1, max_iter+1):
            Fint, K, ep_new, alpha_new, vm_elem = self.assemble(u, ep_trial, alpha_trial)
            R = Fint - self.F_ext
            R[self.fixed] = 0.0

            rn = np.linalg.norm(R[free])
            if rn < tol:
                # accept state
                self.epdev = ep_new
                self.alpha = alpha_new
                return u, it, rn, vm_elem

            K_ff = K[free[:,None], free]
            du = np.zeros_like(u)
            du_f = spsolve(K_ff, -R[free])
            du[free] = du_f

            # basic damping (helps near yield)
            damp = 0.8  # start conservative: 0.1~0.3
            u += damp * du

            # update trial state
            ep_trial = ep_new
            alpha_trial = alpha_new

        raise RuntimeError(f"Newton did not converge, last ||R||={rn:e}")

# ============================================================
# Demo: plate in tension, elastoplastic
# ============================================================
E = 210e9
nu = 0.30
t  = 0.01

sigy0 = 250e6   # yield stress
H     = 1.0e9   # hardening modulus (keep >0 for stability)

Lx, Ly = 1.0, 0.2
nx, ny = 80, 16

nodes, elems = make_rect_mesh_Q4(Lx, Ly, nx, ny)

solver = Q4PlaneStrainJ2Solver(nodes, elems, E, nu, t, sigy0, H)
solver.fix_left_edge(nx, ny)

# Load stepping
Tx_max = 250e6   # traction (Pa)
nsteps = 100
loads = np.linspace(0.0, Tx_max, nsteps)

u = np.zeros(2*nodes.shape[0], dtype=float)
vm_hist = []
ux_tip = []

for k, Tx in enumerate(loads):
    solver.set_traction_right_edge(Lx, Ly, nx, ny, Tx)
    u, nit, rn, vm_elem = solver.solve_loadstep(u, max_iter=40, tol=1e-7)
    vm_hist.append(vm_elem.max())
    # tip displacement (top-right node)
    tip = ny*(nx+1) + nx
    ux_tip.append(u[2*tip])

    print(f"Step {k+1}/{nsteps}: Tx={Tx/1e6:.1f} MPa | iters={nit} | max_vm={vm_elem.max()/1e6:.1f} MPa | tip ux={u[2*tip]:.3e}")

# Plot traction-displacement
plt.figure()
plt.plot(np.array(ux_tip), loads/1e6)
plt.xlabel("Tip ux (m)")
plt.ylabel("Traction Tx (MPa)")
plt.title("Elastoplastic response (plane strain J2)")
plt.grid(True)
plt.show()

# Plot final von Mises field (element avg) on deformed mesh
scale = 50
xy_def = nodes.copy()
xy_def[:,0] += scale*u[0::2]
xy_def[:,1] += scale*u[1::2]

polys = [xy_def[conn] for conn in elems]
pc = PolyCollection(polys, array=vm_elem, edgecolors='k', linewidths=0.2)
plt.figure(figsize=(9,3))
plt.gca().add_collection(pc)
plt.colorbar(pc, label="von Mises (Pa) [element avg]")
plt.axis('equal')
plt.title("Plane strain J2 plasticity: von Mises on deformed mesh (scaled)")
plt.show()
