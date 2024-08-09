#!/usr/bin/env python

# Rewrite the QED-CIS part to match Foley Lab's tutorial QED-CIS-1
# https://github.com/FoleyLab/psi4polaritonic/blob/cpr/QED-CIS-1.ipynb

import numpy as np
from pyscf import gto, scf
import pandas as pd
from scipy.linalg import fractional_matrix_power

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

# Set numpy print options: 8 decimal places, suppress small numbers
np.set_printoptions(precision=8, suppress=True)

def run_qed_hf(mymf, lambda_cav):
    """QED-HF
    TODO: Reduce the number of return values as some are redundant

    Args:
        mymf (pyscf.scf.hf.RHF): PySCF RHF object
        lambda_cav (np.array): Cavity coupling strength

    Returns:
        float: QED-HF energy
        np.array: Eigenvalues
        np.array: MO coefficients
        np.array: Density matrix
        float: x-component of electronic dipole moment
        float: y-component of electronic dipole moment
        float: z-component of electronic dipole moment
        np.array: 2-electron integrals
        np.array: AO dipole integrals
    """
    mol = mymf.mol
    E_threshold = mymf.conv_tol

    #
    # SCF initialization
    #
    S = mymf.get_ovlp()
    eri_ao = mol.intor("int2e")
    A = fractional_matrix_power(S, -0.5)
    A = np.asarray(A)

    # Build core Hamiltonian
    H_core = mymf.get_hcore()

    # nuclear-neculear repulsion energy
    E_nuc = mol.energy_nuc()

    # guess density matrix
    D = mymf.make_rdm1()

    # coupling strength
    lambda_x = lambda_cav[0]
    lambda_y = lambda_cav[1]
    lambda_z = lambda_cav[2]

    #
    # photonic initialization
    #

    # electronic dipole integrals in AO basis
    nao = mol.nao
    dip_ao = -1.0 * mol.intor("int1e_r").reshape(3, nao, nao)
    mu_ao_x = dip_ao[0]
    mu_ao_y = dip_ao[1]
    mu_ao_z = dip_ao[2]

    # \lambda \cdot \mu_el (see within the sum of line 3 of Eq. (9) in [McTague:2021:ChemRxiv])
    l_dot_mu_el = lambda_x * mu_ao_x + lambda_y * mu_ao_y + lambda_z * mu_ao_z

    # compute electronic dipole expectation value with
    # canonical RHF density
    mu_exp_el_x = np.einsum("pq, qp ->", mu_ao_x, D, optimize=True)
    mu_exp_el_y = np.einsum("pq, qp ->", mu_ao_y, D, optimize=True)
    mu_exp_el_z = np.einsum("pq, qp ->", mu_ao_z, D, optimize=True)

    # get electronic dipole expectation value
    mu_exp_el = np.array([mu_exp_el_x, mu_exp_el_y, mu_exp_el_z])

    # \lambda \cdot < \mu > where <\mu> contains ONLY electronic contributions
    l_dot_mu_exp_el = np.dot(lambda_cav, mu_exp_el)

    # nuclear dipole
    charges = mol.atom_charges()

    r = mol.atom_coords()  # Assume origin is at (0,0,0)
    nuc_dip = np.einsum("g,gx->x", charges, r)
    nuc_dip_x = nuc_dip[0]
    nuc_dip_y = nuc_dip[1]
    nuc_dip_z = nuc_dip[2]

    l_dot_mu_nuc = (
        lambda_x * nuc_dip_x
        + lambda_y * nuc_dip_y
        + lambda_z * nuc_dip_z
    )
    # electronic quadrupole
    quadrupole = -1.0 * mol.intor("int1e_rr").reshape(3, 3, nao, nao)
    Q_ao_xx = quadrupole[0, 0]
    Q_ao_xy = quadrupole[0, 1]
    Q_ao_xz = quadrupole[0, 2]
    Q_ao_yy = quadrupole[1, 1]
    Q_ao_yz = quadrupole[1, 2]
    Q_ao_zz = quadrupole[2, 2]

    # Pauli-Fierz 1-e quadrupole terms, Line 2 of Eq. (9) in [McTague:2021:ChemRxiv]
    Q_PF = -0.5 * lambda_x * lambda_x * Q_ao_xx
    Q_PF += -0.5 * lambda_y * lambda_y * Q_ao_yy
    Q_PF += -0.5 * lambda_z * lambda_z * Q_ao_zz

    # accounting for the fact that Q_ij = Q_ji
    # by weighting Q_ij x 2 which cancels factor of 1/2
    Q_PF += -1.0 * lambda_x * lambda_y * Q_ao_xy
    Q_PF += -1.0 * lambda_x * lambda_z * Q_ao_xz
    Q_PF += -1.0 * lambda_y * lambda_z * Q_ao_yz

    # Pauli-Fierz 1-e dipole terms scaled <\mu>_e
    d_PF = -1 * l_dot_mu_exp_el * l_dot_mu_el

    # constant term:  Pauli-Fierz (\lambda \cdot <\mu>_e ) ^ 2
    d_c = 0.5 * l_dot_mu_exp_el * l_dot_mu_exp_el

    # Add Pauli-Fierz terms to H_core
    # Eq. (11) in [McTague:2021:ChemRxiv]
    H0 = H_core + Q_PF + d_PF

    E_1el_crhf = np.einsum("pq,qp->", H_core, D)
    E_1el = np.einsum("pq,qp->", H0, D)
    print("Canonical RHF One-electron energy = %4.16f" % E_1el_crhf)
    print("CQED-RHF One-electron energy      = %4.16f" % E_1el)
    print("Nuclear repulsion energy          = %4.16f" % E_nuc)
    print("Dipole energy                     = %4.16f" % d_c)
    print("\n")

    nocc = np.count_nonzero(mymf.mo_occ > 0)

    E_old = 0
    MAXITER = 50

    # Scale D, J, K by 1/2 to match FoleyLab code
    J, K = mymf.get_jk(dm=D)
    D /= 2.0

    for scf_iter in range(1, MAXITER + 1):

        # J and K should be updated in loop
        J = np.einsum("pqrs,rs->pq", eri_ao, D)
        K = np.einsum("prqs,rs->pq", eri_ao, D)

        # update cavity terms
        mu_exp_el_x = np.einsum("pq, qp ->", mu_ao_x, 2 * D, optimize=True)
        mu_exp_el_y = np.einsum("pq, qp ->", mu_ao_y, 2 * D, optimize=True)
        mu_exp_el_z = np.einsum("pq, qp ->", mu_ao_z, 2 * D, optimize=True)
        # \lambda \cdot < \mu > where <\mu> contains ONLY electronic contributions
        l_dot_mu_exp_el = np.dot(lambda_cav, mu_exp_el)
        # Pauli-Fierz 1-e dipole terms scaled <\mu>_e
        d_PF = -1 * l_dot_mu_exp_el * l_dot_mu_el
        # constant terms:  0.5 ( lambda . ( dn - <d> ) )^2 = 0.5 ( lambda . <de> )^2
        d_c = 0.5 * l_dot_mu_exp_el * l_dot_mu_exp_el
        # Update core Hamiltonian
        H = H_core + Q_PF + d_PF
        # Pauli-Fierz 2-e dipole-dipole terms, line 2 of Eq. (12) in [McTague:2021:ChemRxiv]
        # two-electron part of e-e term (J)
        M = np.einsum(
            "pq,rs,sr->pq",
            l_dot_mu_el,
            l_dot_mu_el,
            D,
            optimize=True,
        )

        N = np.einsum(
            "pr,qs,sr->pq",
            l_dot_mu_el,
            l_dot_mu_el,
            D,
            optimize=True,
        )
        # Update fock matrix by adding plus Pauli-Fierz terms Eq. (12) in [McTague:2021:ChemRxiv]
        # Dipole self energy (DSE) Fock Matrix
        F = H + 2 * J - K + 2 * M - N

        E_new = np.einsum("pq,qp->", (H0 + F), D, optimize=True) + E_nuc + d_c

        print(
            "SCF Iteration %3d: Energy = %4.16f dE = % 1.5E"
            % (scf_iter, E_new, E_new - E_old)
        )

        # SCF Converged?
        if abs(E_new - E_old) < E_threshold:
            break
        
        E_old = E_new
        
        Fp = A.dot(F).dot(A)
        e, C2 = np.linalg.eigh(Fp)
        C = A.dot(C2)
        Cocc = C[:, :nocc]

        D = np.einsum("pi,qi->pq", Cocc, Cocc)  # [Szabo:1996] Eqn. 3.145, pp. 139
        
        # update electronic dipole expectation value
        mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
        mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
        mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

        mu_exp_x += nuc_dip_x
        mu_exp_y += nuc_dip_y
        mu_exp_z += nuc_dip_z
        

        # update \lambda \cdot <\mu>
        l_dot_mu_exp = (
            lambda_x * mu_exp_x
            + lambda_x * mu_exp_y
            + lambda_x * mu_exp_z
        )
        
        # Line 3 in full of Eq. (9) in [McTague:2021:ChemRxiv]
        d_PF = (l_dot_mu_nuc - l_dot_mu_exp) * l_dot_mu_el

        # update Core Hamiltonian
        H = H_core + Q_PF + d_PF

        # update dipole energetic contribution, Eq. (14) in [McTague:2021:ChemRxiv]
        d_c = (
            0.5 * l_dot_mu_nuc ** 2
            - l_dot_mu_nuc * l_dot_mu_exp
            + 0.5 * l_dot_mu_exp ** 2
        )

    print("QED-RHF   energy: %.8f hartree" % E_new)

    return E_new, e, C, D, mu_exp_x, mu_exp_y, mu_exp_z, eri_ao, dip_ao


def run_qed_cis(mymf, lambda_cav, cavity_frequency, e, C, D, mu_exp_x, mu_exp_y, mu_exp_z, eri_ao, dip_ao):
    """QED-CIS
    TODO: Reduce the number of input arguments as some are redundant
    
    Args:
        mymf (pyscf.scf.hf.RHF): PySCF RHF object
        lambda_cav (np.array): Cavity coupling strength
        cavity_frequency (float): Cavity frequency
        e (np.array): Eigenvalues
        C (np.array): MO coefficients
        D (np.array): Density matrix
        mu_exp_x (float): x-component of electronic dipole expectation value
        mu_exp_y (float): y-component of electronic dipole expectation value
        mu_exp_z (float): z-component of electronic dipole expectation value
        eri_ao (np.array): 2-electron integrals
        dip_ao (np.array): dipole integrals in AO basis

    Returns:
        np.array: Eigenvalues
        np.array: MO coefficients
    """

    nmo = mymf.mo_occ.size
    nocc = np.count_nonzero(mymf.mo_occ > 0)
    nvir = nmo - nocc

    print(nocc, nmo, nvir)

    lambda_x = lambda_cav[0]
    lambda_y = lambda_cav[1]
    lambda_z = lambda_cav[2]

    mu_ao_x = dip_ao[0]
    mu_ao_y = dip_ao[1]
    mu_ao_z = dip_ao[2]

    cqed_rhf_dipole_moment = np.array([mu_exp_x, mu_exp_y, mu_exp_z])

    mo_coeff = mymf.mo_coeff

    eps = e
    mo_energy_o = eps[:nocc]
    mo_energy_v = eps[nocc:]
    print(f"eps_o: {mo_energy_o}")
    print(f"eps_v: {mo_energy_v}")
    print(f"Density matrix: {D}")

    mo_coeff = C
    get_eris = np.einsum("uvkl, up, vq, kr, ls -> pqrs", eri_ao, mo_coeff, mo_coeff, mo_coeff, mo_coeff, optimize=True)
    ovov = get_eris[:nocc,nocc:,:nocc,nocc:].copy()
    oovv = get_eris[:nocc,:nocc,nocc:,nocc:].copy()

    # print(f"ovov: {ovov}")
    # print(f"oovv: {oovv}")

    mu_cmo_x = np.dot(C.T, mu_ao_x).dot(C)
    mu_cmo_y = np.dot(C.T, mu_ao_y).dot(C)
    mu_cmo_z = np.dot(C.T, mu_ao_z).dot(C)

    l_dot_mu_exp = 0.0
    for i in range(0, 3):
        l_dot_mu_exp += lambda_cav[i] * cqed_rhf_dipole_moment[i]

    l_dot_mu_el = lambda_x * mu_cmo_x
    l_dot_mu_el += lambda_y * mu_cmo_y
    l_dot_mu_el += lambda_z * mu_cmo_z

    #
    # The remaining of this function is adapted from FoleyLab's QED-CIS-1
    # https://github.com/FoleyLab/psi4polaritonic/blob/cpr/QED-CIS-1.ipynb
    #

    # rename some variables for to match QED-CIS-1.ipynb
    ndocc = nocc
    nvirt = nvir
    omega_val = cavity_frequency
    d_el = l_dot_mu_el
    eps_v = mo_energy_v
    eps_o = mo_energy_o

    # build g matrix and its adjoint
    g = np.zeros((1,ndocc * nvirt))
    g_dag = np.zeros((ndocc * nvirt, 1))
    for i in range(0, ndocc):
        for a in range(0, nvirt):
            A = a + ndocc
            ia = i * nvirt + a 
            g[0,ia] = (
                -np.sqrt(omega_val) * d_el[i, A]
            )

    # Now compute the adjoint of g
    g_dag = np.conj(g).T
    #print(g_dag)

    # A
    A_matrix = np.zeros((ndocc * nvirt, ndocc * nvirt))
    # Delta
    D_matrix = np.zeros((ndocc * nvirt, ndocc * nvirt))
    # G
    G = np.zeros((ndocc * nvirt, ndocc * nvirt))
    # \Omega
    Omega = np.zeros((ndocc * nvirt, ndocc * nvirt))

    for i in range(0, ndocc):
        for a in range(0, nvirt):
            A = a + ndocc
            ia = i * nvirt + a
            for j in range(0, ndocc):
                for b in range(0, nvirt):
                    B = b + ndocc
                    jb = j * nvirt + b
                    
                    # ERI contribution to A + \Delta
                    A_matrix[ia, jb] = (2.0 * ovov[i, a, j, b] - oovv[i, j, a, b])
                    
                    # 2-electron dipole contribution to A + \Delta
                    D_matrix[ia, jb] += 2.0 * d_el[i, A] * d_el[j, B]
                    D_matrix[ia, jb] -= d_el[i, j] * d_el[A, B]
                    
                    # bilinear coupling contributions to G
                    # off-diagonal terms (plus occasional diagonal terms)
                    G[ia, jb] += np.sqrt(omega_val / 2) * d_el[i, j] * (a == b)
                    G[ia, jb] -= np.sqrt(omega_val / 2) * d_el[A, B] * (i == j)
                    
                    # diagonal contributions to A_p_D, G, and \Omega matrix
                    if i == j and a == b:
                        # orbital energy contribution to A + \Delta ... this also includes 
                        # the DSE terms that contributed to the CQED-RHF energy 
                        A_matrix[ia, jb] += eps_v[a]
                        A_matrix[ia, jb] -= eps_o[i] 
                        
                        # diagonal \omega term
                        Omega[ia, jb] = omega_val
                        
    # define the offsets
    R0_offset = 0
    S0_offset = 1
    R1_offset = ndocc * nvirt + 1
    S1_offset = ndocc * nvirt + 2

    # CISS Hamiltonians
    H_QED_CIS_1 = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2))
    H_JC_CIS_1 = np.zeros((ndocc * nvirt * 2 + 2, ndocc * nvirt * 2 + 2))

    # build the supermatrix
    # g coupling
    # QED-CIS-1
    H_QED_CIS_1[R0_offset:S0_offset, S1_offset:] = g
    H_QED_CIS_1[S0_offset:R1_offset, R1_offset:S1_offset] = g_dag
    H_QED_CIS_1[R1_offset:S1_offset, S0_offset:R1_offset] = g 
    H_QED_CIS_1[S1_offset:,          R0_offset:S0_offset] = g_dag 
    # JC-CIS-1
    H_JC_CIS_1[R0_offset:S0_offset, S1_offset:] = g
    H_JC_CIS_1[S0_offset:R1_offset, R1_offset:S1_offset] = g_dag
    H_JC_CIS_1[R1_offset:S1_offset, S0_offset:R1_offset] = g
    H_JC_CIS_1[S1_offset:,          R0_offset:S0_offset] = g_dag

    # A + \Delta for QED-CIS-1
    H_QED_CIS_1[S0_offset:R1_offset, S0_offset:R1_offset] = A_matrix + D_matrix

    # A for JC
    H_JC_CIS_1[S0_offset:R1_offset, S0_offset:R1_offset] = A_matrix

    # omega
    # QED-CIS-1
    H_QED_CIS_1[R1_offset, R1_offset] = omega_val
    # JC-CIS-1
    H_JC_CIS_1[R1_offset, R1_offset] = omega_val

    # A + \Delta + \Omega for QED-CIS-1
    H_QED_CIS_1[S1_offset:, S1_offset:] = A_matrix + D_matrix + Omega

    # A + \Omega for JC-CIS-1
    H_JC_CIS_1[S1_offset:, S1_offset:] = A_matrix + Omega

    # G coupling
    # QED-CIS-1
    H_QED_CIS_1[S1_offset:,S0_offset:R1_offset] = G 
    H_QED_CIS_1[S0_offset:R1_offset, S1_offset:] = G 
    # JC-CIS-1
    H_JC_CIS_1[S1_offset:,S0_offset:R1_offset] = G
    H_JC_CIS_1[S0_offset:R1_offset, S1_offset:] = G

    # define the CIS offsets
    CIS_S0_offset = 0
    CIS_R1_offset = ndocc * nvirt

    # CIS Hamiltonians
    H_QED_CIS = np.zeros((ndocc * nvirt + 1, ndocc * nvirt + 1))
    H_JC_CIS = np.zeros((ndocc * nvirt + 1, ndocc * nvirt + 1))

    # build the supermatrix
    # g coupling
    # QED-CIS
    H_QED_CIS[CIS_R1_offset:, CIS_S0_offset:CIS_R1_offset] = g 
    H_QED_CIS[CIS_S0_offset:CIS_R1_offset, CIS_R1_offset:] = g_dag 
    # JC-CIS
    H_JC_CIS[CIS_R1_offset:, CIS_S0_offset:CIS_R1_offset] = g
    H_JC_CIS[CIS_S0_offset:CIS_R1_offset, CIS_R1_offset:] = g_dag 

    # A + \Delta for QED-CIS
    H_QED_CIS[CIS_S0_offset:CIS_R1_offset, CIS_S0_offset:CIS_R1_offset] = A_matrix + D_matrix
    # A  for JC-CIS
    H_JC_CIS[CIS_S0_offset:CIS_R1_offset, CIS_S0_offset:CIS_R1_offset] = A_matrix

    # omega
    # QED-CIS
    H_QED_CIS[CIS_R1_offset, CIS_R1_offset] = omega_val
    # JC-CIS
    H_QED_CIS[CIS_R1_offset, CIS_R1_offset] = omega_val

    # diagonalize the QED-CIS-1 matrix!
    # each eigenvector is ordered as [C_00^0, C_ia^0, C_00^1, C_ia^1]
    E_QED_CIS_1, C_QED_CIS_1 = np.linalg.eigh(H_QED_CIS_1)

    # (optional) diagonalize other matrices that might be useful later
    E_JC_CIS_1, C_JC_CIS_1 = np.linalg.eigh(H_JC_CIS_1)
    E_QED_CIS, C_QED_CIS = np.linalg.eigh(H_QED_CIS)
    E_JC_CIS, C_JC_CIS = np.linalg.eigh(H_JC_CIS)

    return E_QED_CIS_1, C_QED_CIS_1


def run_e_weight(mymf, ECIS, L_CCIS):
    """Compute the electronic weight of each state

    Args:
        mymf (pyscf.scf.hf.RHF): PySCF RHF object
        ECIS (np.array): Eigenvalues directly from QED-CIS solver
        L_CCIS (np.array): CI coefficients directly from QED-CIS solver

    Returns:
        np.array: Electronic weight of each state
    """
    nmo = mymf.mo_occ.size
    nocc = np.count_nonzero(mymf.mo_occ > 0)
    nvir = nmo - nocc
    nov = nocc * nvir

    # Make copies of CI energies and coefficients to avoid modifying the original arrays
    ci_energies = np.copy(ECIS)
    ci_coeffs = np.copy(L_CCIS)
    # For all states, get the first 1 to 1+nocc*nvir coefficients, since they correspond to C_ia^0 (the coefficients afterwards correspond to C_00^1 and C_ia^1)
    cia_0 = ci_coeffs[1:1+nov, :]

    nstates = len(ci_energies)
    weight_es = np.zeros(nstates)

    for n in range(nstates):
        # sum the squares of C_ia^0 for each state
        sum_sq_0 = np.sum(np.abs(cia_0[:, n])**2)
        # sum the squares of all C's for each state
        sum_sq_total = np.sum(np.abs(ci_coeffs[:, n])**2)
        # weight equals sum_sq_0 divided by sum_sq_total
        weight_es[n] = sum_sq_0 / sum_sq_total
    
    return weight_es


def plot_electronic_weights(weights, fig_name, cmap=pl.cm.Reds, xlim=(0, 50), ylim=(0, 50)):
    """Plot a heatmap of electronic weights as a function of cavity frequency (x axis) and excitation energy (y axis). Save the figure to a file.

    Args:
        weights (pd.DataFrame): Dataframe containing columns "cavity_frequency", "excitation_energy", and "electronic_weight"
        fig_name (str): Name of the figure file

    Returns:
        None
    """
    # Add transparency to the colormap
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set tranaparency alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    # create a figure of size 4x4 inches, 300 dots per inch
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111)
    heatmap = ax.scatter(weights["cavity_frequency"], weights["excitation_energy"], c=weights["electronic_weight"], s=10, marker="_", cmap=my_cmap)
    cbar = plt.colorbar(heatmap)

    # setting labels and limits
    cbar.set_label("Electronic Weight")
    ax.set_xlabel("Cavity Frequency (eV)")
    ax.set_ylabel("Excitation Energy (eV)")
    ax.set_ylim(xlim)
    ax.set_xlim(ylim)

    # save figure
    plt.savefig(fig_name)

def run_spectral(ECIS, L_CCIS, w, eta):
    """ Compute the spectral function of each state   
    
    Args:
        mymf (pyscf.scf.hf.RHF): PySCF RHF object
        ECIS (np.array): Eigenvalues directly from QED-CIS solver
        L_CCIS (np.array): CI coefficients directly from QED-CIS solver

    Return:
        np.array: spectral function of each state
    """
    ci_eigvals = ECIS
    ci_eigvecs = L_CCIS
    
    #nstates = nocc * nvir
    e0 = 0.0
    en = ci_eigvals
    cn = ci_eigvecs # shape (nstates, nocc, nvir)

    A_w = 0.0
    for ei, ci in zip(en, cn):
        # sum all elements of a CI matrix
        td = np.sum(ci)
        num = np.linalg.norm(td)**2
        denom = w - (ei - e0) + 1j * eta
        A_w += num / denom

    A_w = -1./np.pi * A_w.imag

    return A_w

def plot_spectral_function(spectral, fig_name, cmap=pl.cm.Reds, xlim=(0, 50), ylim=(0, 50)):
    """Plot a heatmap of spectral function as a function of cavity frequency (x axis) and excitation energy (y axis). Save the figure to a file.

    Args:
        spectral_function (np.array): Spectral function of each state
        fig_name (str): Name of the figure file

    Returns:
        None
    """

    # Add transparency to the colormap
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set tranaparency alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    # create a figure of size 4x4 inches, 300 dots per inch
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111)
    heatmap = ax.scatter(spectral["cavity_frequency"], spectral["excitation_energy"], c=spectral["spectral_function"], s=10, marker="_", cmap=my_cmap)
    cbar = plt.colorbar(heatmap)

    # setting labels and limits
    cbar.set_label("Electronic Weight")
    ax.set_xlabel("Cavity Frequency (eV)")
    ax.set_ylabel("Excitation Energy (eV)")
    ax.set_ylim(xlim)
    ax.set_xlim(ylim)

    # save figure
    plt.savefig(fig_name)

    # setting labels and limits
    cbar.set_label("Spectral Function")
    ax.set_xlabel("Cavity Frequency (eV)")
    ax.set_ylabel("Excitation Energy (eV)")
    ax.set_ylim(xlim)
    ax.set_xlim(ylim)

    # save figure
    plt.savefig(fig_name)

def run_abs_spec(mymf, ECIS, L_CCIS):
    nao = mymf.mol.nao # For STO-3G H2o, Use to be 7, now 7*2
    nmo = mymf.mo_occ.size
    nocc = np.count_nonzero(mymf.mo_occ > 0)
    nvir = nmo - nocc
    nov = nocc * nvir
    
    # Make copies of CI energies and coefficients to avoid modifying the original arrays
    ci_coeffs = np.copy(L_CCIS)
    # For all states, get the first 1 to 1+nocc*nvir coefficients, since they correspond to C_ia^0 (the coefficients afterwards correspond to C_00^1 and C_ia^1)
    cia_0 = ci_coeffs[1:1+nov, :]
    
    td_x = cia_0.T[:, :nocc*nvir]
    X_ia = td_x.reshape(nocc*nvir*2+2, nocc, nvir)
    
    # For STO-3G H2o, Use to be (3,7,7), now (3,7*2,7*2)
    dip_ao = -1.0 * mymf.mol.intor("int1e_r").reshape(3, nao, nao)
    so, sv = slice(0, nocc), slice(nocc, nmo)

    C = mymf.mo_coeff
    d_mo = np.einsum("up, tuv, vq -> tpq", C, dip_ao, C)
    d_ia = d_mo[:, so, sv]
        
    td_transdip = np.einsum("tia, nia -> nt", d_ia, X_ia)
    fosc = td_transdip[:]**2
    fosc = 4./3. * ECIS * fosc.sum(axis=1)

    return fosc

def plot_abs_spec(abs_spec, fig_name, cmap=pl.cm.Reds, xlim=(0, 50), ylim=(0, 50)):
    """Plot a heatmap of absorption spectrum as a
    function of cavity frequency (x axis) 
    and excitation energy (y axis). 
    Save the figure to a file.
    
    Args:
        abs_spec (np.array): Absorption spectrum
        fig_name (str): Name of the figure file
        
    Returns:
        None
    """
    # Add transparency to the colormap
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set tranaparency alpha
    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    # create a figure of size 4x4 inches, 300 dots per inch
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(111)
    heatmap = ax.scatter(abs_spec["cavity_frequency"], abs_spec["excitation_energy"], c=abs_spec["absorption_spectrum"], s=10, marker="_", cmap=my_cmap)
    cbar = plt.colorbar(heatmap)

    # setting labels and limits
    cbar.set_label("Absorption Spectrum")
    ax.set_xlabel("Cavity Frequency (eV)")
    ax.set_ylabel("Excitation Energy (eV)")
    ax.set_ylim(xlim)
    ax.set_xlim(ylim)

    # save figure
    plt.savefig(fig_name)
        
def run_oscillator_strength(mymf, mo_coeff, ECIS, L_CCIS, dip_ao):
    """Computes the oscillator strength for a set of electronic states.

    Args:
        mymf (pyscf.scf.hf.RHF): PySCF RHF object
        mo_coeff (np.array): MO coefficients from QED-HF
        ECIS (np.array): Eigenvalues directly from QED-CIS solver
        L_CCIS (np.array): CI coefficients directly from QED-CIS solver
        dip_ao (np.array): Dipole integrals in AO basis

    Returns:
        np.array: Oscillator strength
    """ 

    nmo = mymf.mo_occ.size
    nocc = np.count_nonzero(mymf.mo_occ > 0)
    nvir = nmo - nocc
    nov = nocc * nvir

    # Make copies of CI energies and coefficients to avoid modifying the original arrays
    ci_energies = np.copy(ECIS)
    ci_coeffs = np.copy(L_CCIS)
    
    # For all states, get the first 1 to 1+nocc*nvir coefficients, since they correspond to C_ia^0 (the coefficients afterwards correspond to C_00^1 and C_ia^1)
    cia_0 = ci_coeffs[1:1+nov, :]

    nstates = len(ci_energies)
    cia_0 = cia_0.T.reshape(nstates, nocc, nvir) 

    # Convert dipole matrices from AO to MO basis
    d_mo = np.einsum("up, tuv, vq -> tpq", mo_coeff, dip_ao, mo_coeff)
    # Get occ-vir block of dipole 
    d_ia = d_mo[:, :nocc, nocc:]

    td_transdip = np.einsum("tia, nia -> nt", d_ia, cia_0)

    fosc = np.abs(td_transdip[:])**2
    fosc = 4./3. * ci_energies * fosc.sum(axis=1)

    return fosc

if __name__ == "__main__":

    # Build molecule
    mol = gto.Mole()
    mol.unit = "A"
    mol.atom = """
    H    3.361137    -1.238078    0.000000
    C    2.411654    -0.699292    0.000000
    H    1.237065    -2.480854    0.000000
    C    1.232905    -1.388655    0.000000
    C    1.232905    1.388655    0.000000
    C    0.000000    -0.712406    0.000000
    C    2.411654    0.699292    0.000000
    C    0.000000    0.712406    0.000000
    C   -1.232905    -1.388655    0.000000
    H    3.361137    1.238078    0.000000
    H   -1.237065    2.480854    0.000000
    H    1.237065    2.480854    0.000000
    C   -2.411654    -0.699292    0.000000
    H   -1.237065    -2.480854    0.000000
    H   -3.361137    -1.238078    0.000000
    C   -2.411654    0.699292    0.000000
    H   -3.361137    1.238078    0.000000
    C   -1.232905    1.388655    0.000000
    """
    mol.basis = "ccpvdz"
    mol.verbose = 5
    mol.charge = 0
    mol.build()

    # Set SCF convergence options
    E_threshold = 1e-10
    G_threshold = 1e-10

    # Run regular HF
    mymf = scf.RHF(mol)
    mymf.conv_tol = E_threshold
    mymf.conv_tol_grad = G_threshold
    divider = f"{''.join(['*']*8)} Running regular HF {''.join(['*']*8)}"
    print(f"\n{divider}\n")
    ehf = mymf.kernel()

    au2ev = 27.211386245988
    # Set QED options
    # cavity frequency
    cavity_frequency = 0 # eV
    cavity_frequency /= au2ev
    # coupling strength
    lambda_cav = np.array([0.01, 0.0, 0.0])

    # Run QED-HF
    divider = f"{''.join(['*']*8)} Running QED-HF {''.join(['*']*8)}"
    print(f"\n{divider}\n")
    E_new, e, C, D, mu_exp_x, mu_exp_y, mu_exp_z, eri_ao, dip_ao  = run_qed_hf(mymf, lambda_cav)

    # Run QED-CIS
    divider = f"{''.join(['*']*8)} Running QED-CIS {''.join(['*']*8)}"
    print(f"\n{divider}\n")
    ECIS, L_CCIS = run_qed_cis(mymf, lambda_cav, cavity_frequency, e, C, D, mu_exp_x, mu_exp_y, mu_exp_z, eri_ao, dip_ao)
    print(f"\nState energies (in eV): {ECIS*au2ev}")
    print(f"Excitation energies (in ev): {(ECIS - ECIS[0])*au2ev}")
    
    # Run QED-CIS oscillator strength
    fosc = run_oscillator_strength(mymf, C, ECIS, L_CCIS, dip_ao)

    # Save excitation energies (eV) and oscillator strengths in pd.DataFrame for printing (or plotting)
    df = pd.DataFrame({
        "Energy (eV)": (ECIS - ECIS[0])*au2ev,
        "fosc": fosc
    })
    print(df[:22])
    #exit(1)
    #
    # QED-CIS electron weight
    #
    divider = f"{''.join(['*']*8)} Running QED-CIS Electron Weight {''.join(['*']*8)}"
    print(f"\n{divider}\n")

    # scan cavity freqeuency from 0 to 50 eV, with 100 steps
    ws = np.linspace(0, 50, 100)
    # create a pandas dataframe to store "cavity frequency", "excitation energy", and "electronic weight"
    all_weights = pd.DataFrame(columns=["cavity_frequency", "excitation_energy", "electronic_weight"])
    all_spectral = pd.DataFrame(columns=["cavity_frequency", "excitation_energy", "spectral_function"])
    all_abs_spec = pd.DataFrame(columns=["cavity_frequency", "excitation_energy", "absorption_spectrum"])   

    # Since QED-HF is independent of cavity frequency, we do not have to rerun it for each cavity frequency.
    # Loop over all cavity frequencies, and for each cavity frequency, we run QED-CIS and compute the electronic weight
    for cavity_frequency in ws:
        ECIS, L_CCIS = run_qed_cis(mymf, lambda_cav, cavity_frequency/au2ev, e, C, D, mu_exp_x, mu_exp_y, mu_exp_z, eri_ao, dip_ao)
        weight = run_e_weight(mymf, ECIS, L_CCIS)
        
        # # generate a range of frequencies from 0 to 50 eV with 1000 points
        # ws = np.linspace(0, 50, 1000)

        # # broadening parameter
        # eta = 0.01 

        # # compute the spectral function
        # spectrum = np.zeros_like(ws)
        # for i, wi in enumerate(ws):
        #     spectrum[i] = compute_spectral_function(ECIS, L_CCIS_for_abs, wi/au2ev, eta)
        
        ws = np.linspace(0, 50, 100)
        eta = 0.01 
        spectral = np.zeros_like(ws) 
        for i, wi in enumerate(ws):
            spectral = run_spectral(ECIS, L_CCIS, wi, eta)
        
        abs_spec = run_oscillator_strength(mymf, C, ECIS, L_CCIS, dip_ao)    
        #abs_spec = run_abs_spec(mymf, ECIS, L_CCIS)
        # create a dataframe to store the electronic weight
        weight_df = pd.DataFrame({"cavity_frequency": cavity_frequency*np.ones(len(ECIS)), "excitation_energy": ECIS*au2ev, "electronic_weight": weight})
        spectral_df = pd.DataFrame({"cavity_frequency": cavity_frequency*np.ones(len(ECIS)), "excitation_energy": ECIS*au2ev, "spectral_function": spectral})
        abs_spec_df = pd.DataFrame({"cavity_frequency": cavity_frequency*np.ones(len(ECIS)), "excitation_energy": ECIS*au2ev, "absorption_spectrum": abs_spec})
        # append the dataframe to the all_weights dataframe
        all_weights = pd.concat([all_weights, weight_df])
        all_spectral = pd.concat([all_spectral, spectral_df])
        all_abs_spec = pd.concat([all_abs_spec, abs_spec_df])

    # save the all_weights dataframe to a csv file so that we can adjust plot later without rerunning calculations
    all_weights.to_csv("data-electronic_weight.csv", index=False)
    all_spectral.to_csv("data-spectral_function.csv", index=False)
    all_abs_spec.to_csv("data-absorption_spectrum.csv", index=False)

    # plot the electronic weight
    plot_electronic_weights(all_weights, "./fig-electronic_weight.pdf")

    # plot the spectral function
    plot_spectral_function(all_spectral, "./fig-spectral_function.pdf")

    # plot the absorption spectrum
    plot_abs_spec(all_abs_spec, "./fig-absorption_spectrum.pdf")

