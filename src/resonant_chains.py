import rebound as rb
import numpy as np
import celmech as cm
import sympy as sp
from celmech.canonical_transformations import CanonicalTransformation
from celmech.nbody_simulation_utilities import add_canonical_heliocentric_elements_particle as add_ch_particle
def get_chain_hamiltonian(planet_masses,planet_resonances):
    sim = rb.Simulation()
    sim.add(m=1)
    Npl = len(planet_masses)
    Period = 1
    for i,m in enumerate(planet_masses):
        add_ch_particle(m,{'P':Period},sim)
        if i<Npl-1:
            j,k = planet_resonances[i]
            Period *= j / (j-k)
    sim.move_to_com()
    pvars = cm.Poincare.from_Simulation(sim)
    pham = cm.PoincareHamiltonian(pvars)
    L0s = pham.Lambda0s[1:]
    Ls = pham.qp_vars[pham.N_dof::3]
    L2L0 = dict(zip(Ls,L0s))
    nvec = [n.xreplace(L2L0) for n in pham.flow[:pham.N_dof:3]]

    # subtract Keplerian piece, we will add it back later
    Hkep = pham.H
    pham.H -= Hkep
    # add resonant terms
    for i,jk in enumerate(planet_resonances):
        j,k = jk
        pham.add_MMR_terms(j,k,indexIn=i+1,indexOut = i+2)
    for i in range(1,Npl+1):
        for j in range(i+1,Npl+1):
            pham.add_secular_terms(indexIn = i, indexOut = j)
    
    A = resonances_to_angle_transormation_matrix(planet_resonances)
    Qvars = sp.symbols("phi(1:{0}),psi,h,r(1:{1}),s(1:{1})".format(Npl-1,Npl+1)) 
    Pvars = sp.symbols("Phi(1:{0}),Psi,H,R(1:{1}),S(1:{1})".format(Npl-1,Npl+1)) 
    QPvars = list(zip(Qvars,Pvars))
    ct = CanonicalTransformation.from_poincare_angles_matrix(
       pham.state,
       A,
       new_qp_pairs=QPvars
    )

    kam = ct.old_to_new_hamiltonian(pham)

    new_cart_vars = list(zip(sp.symbols("x(1:{0}),u(1:{0})".format(Npl+1)),sp.symbols("y(1:{0}),v(1:{0})".format(Npl+1))))
    cart_indices = list(range(Npl,3*Npl+1))
    ct_p2xy = CanonicalTransformation.polar_to_cartesian(
        ct.new_qp_vars,
        indices=cart_indices,
        cartesian_symbol_pairs=new_cart_vars
    )
    
    # Re-scale variables
    kam_xy = ct_p2xy.old_to_new_hamiltonian(kam)
    ct_rescale = CanonicalTransformation.rescale_transformation(
        kam_xy.qp_pairs,
        1/pham.Lambda0s[1],
        cartesian_pairs=cart_indices,
        params={pham.Lambda0s[1]:pham.H_params[pham.Lambda0s[1]]}
    )
    kam_xy_rescale = ct_rescale.old_to_new_hamiltonian(kam_xy)
    ct_composite = CanonicalTransformation.composite([ct,ct_p2xy,ct_rescale])
    P0s = []
    for q,p in kam_xy_rescale.qp_pairs[:Npl]:
        P0s.append(sp.Add(*[sp.diff(ct_composite.new_to_old(p),L)*L0 for L,L0 in zip(Ls,L0s)]))

    # expand about Lambda0s
    actions = kam_xy_rescale.qp_vars[kam_xy_rescale.N_dof:kam_xy_rescale.N_dof+Npl]
    ct_dPhi = CanonicalTransformation.actions_to_delta_actions(
        kam_xy_rescale.qp_vars,
        actions,
        [cm.get_symbol("\delta\Phi",i) for i in range(1,Npl+1)],
        actions_ref=P0s,
        params={L0:pham.H_params[L0] for L0 in L0s}
    )

    kam_xy_rescale_dPhi = ct_dPhi.old_to_new_hamiltonian(kam_xy_rescale)
    ct_composite = CanonicalTransformation.composite([ct,ct_p2xy,ct_rescale,ct_dPhi])

    # Keplerian piece of the Hamiltonian, written in transformed variables
    Hkep_new = sp.Add(*[-3 * (ni / sp.S(2) / L0 / L0s[0]) * sp.simplify(ct_composite.old_to_new(L-L0))**2 for ni,L0,L in zip(nvec,L0s,Ls)])
    kam_xy_rescale_dPhi.H += Hkep_new

    return kam_xy_rescale_dPhi, ct_composite, pham

def resonances_to_angle_transormation_matrix(planet_resonances):
    Npl = len(planet_resonances)+1
    Ndof = 3 * Npl
    A = np.eye(Ndof,dtype=int)
    for i in range(Npl-2):
        ji,ki = planet_resonances[i]
        ji1,ki1 = planet_resonances[i+1]
        A[i,i:i+3] = np.array([ki-ji,ji,0]) - np.array([0,ki1-ji1,ji1])
    A[Npl-2,(Npl-2,Npl-1)] = np.array([1,-1])
    jlast,klast = planet_resonances[-1]
    A[Npl-1,(Npl-2,Npl-1)] = jlast - klast,-jlast
    A[Npl:,(Npl-2,Npl-1)]  = klast - jlast, jlast
    return A