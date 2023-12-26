import psi4
import magpy
import pytest
from ..data.molecules import *
import numpy as np
import os

np.set_printoptions(precision=10, linewidth=200, threshold=200, suppress=True)

def test_AAT_HF_H2O():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    psi4.set_options({'basis': 'STO-6G'})
    mol = psi4.geometry(moldict["H2O"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    AAT = magpy.AAT_HF(mol)

    r_disp = 0.0001
    b_disp = 0.0001
    I = AAT.compute(r_disp, b_disp)
    print("\nElectronic Contribution to Atomic Axial Tensor (a.u.):")
    print(I)

    # H2O: SCF/STO-6G electronic contribution to AAT (a.u.)
    analytic_ref = np.array([
    [ 0.00000000000000,    0.00000000000000,   -0.22630653868957],
    [-0.00000000000000,   -0.00000000000000,    0.00000000000000],
    [ 0.32961125700525,   -0.00000000000000,   -0.00000000000000],
    [-0.00000000000000,   -0.00000000000000,    0.05989549730400],
    [ 0.00000000000000,    0.00000000000000,   -0.13650378268362],
    [-0.22920257093325,    0.21587263338256,   -0.00000000000000],
    [-0.00000000000000,   -0.00000000000000,    0.05989549730400],
    [-0.00000000000000,   -0.00000000000000,    0.13650378268362],
    [-0.22920257093325,   -0.21587263338256,    0.00000000000000]
    ])

    print("\nError in Numerical vs. Analytic Tensors (a.u.):")
    print(analytic_ref-I)

    assert(np.max(np.abs(analytic_ref-I)) < 1e-7)

def test_AAT_HF_H2O2():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    psi4.set_options({'basis': 'STO-6G'})
    mol = psi4.geometry(moldict["H2O2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    AAT = magpy.AAT_HF(mol)
    r_disp = 0.0001
    b_disp = 0.0001
    I = AAT.compute(r_disp, b_disp)
    print("\nElectronic Contribution to Atomic Axial Tensor (a.u.):")
    print(I)

    # H2O2: SCF/STO-6G electronic contribution to AAT (a.u.)
    analytic_ref = np.array([
    [-0.17020009136429,    0.08484275371769,    0.64233130671871],
    [-0.09953703423030,    0.00521763971028,    0.11911360152104],
    [-0.44972025759211,   -0.11204113788570,    0.16154272077005],
    [-0.17020009136430,    0.08484275371769,   -0.64233130671871],
    [-0.09953703423030,    0.00521763971028,   -0.11911360152105],
    [ 0.44972025759211,    0.11204113788570,    0.16154272077006],
    [-0.11795348786103,   -0.07664276941919,    0.31734821595902],
    [-0.00351307079869,    0.00418193069258,   -0.04648353083472],
    [-0.16844909726840,    0.21030675621972,    0.12095547794710],
    [-0.11795348786103,   -0.07664276941919,   -0.31734821595903],
    [-0.00351307079869,    0.00418193069258,    0.04648353083472],
    [ 0.16844909726840,   -0.21030675621972,    0.12095547794710]
    ])

    print("\nError in Numerical vs. Analytic Tensors (a.u.):")
    print(analytic_ref-I)

    assert(np.max(np.abs(analytic_ref-I)) < 1e-7)


def test_AAT_HF_etho():

    psi4.set_memory('2 GB')
    psi4.set_output_file('output.dat', False)
    psi4.set_options({'scf_type': 'pk',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-12,
                      'r_convergence': 1e-12})

    os.environ["PSIPATH"] = "magpy/magpy/tests:$PSIPATH"
    psi4.set_options({'basis': '4-31G'})
    mol = psi4.geometry(moldict["Ethylene Oxide"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
    print(f"  SCF Energy from Psi4: {rhf_e}")

    AAT = magpy.AAT_HF(mol)
    r_disp = 0.0001
    b_disp = 0.0001
    I = AAT.compute(r_disp, b_disp)
    print("\nElectronic Contribution to Atomic Axial Tensor (a.u.):")
    print(I)

    # Ethylene oxide: SCF/4-31G electronic contribution to AAT (a.u.)
    analytic_ref = np.array([
    [-0.00000000000001,   -1.30238219256271,    0.00001077548633],
    [ 1.24671563867172,    0.00000000000001,   -0.00000000000000],
    [-0.00000062315155,    0.00000000000000,   -0.00000000000000],
    [-0.00000000000000,    0.67841668368122,   -1.07054644295926],
    [-0.50047243294799,   -0.00000000000001,    0.00000000000002],
    [ 0.66019219382163,    0.00000000000000,    0.00000000000000],
    [ 0.00000000000001,    0.67843176338138,    1.07054705470218],
    [-0.50048410407591,   -0.00000000000001,   -0.00000000000001],
    [-0.66018505657783,    0.00000000000000,    0.00000000000000],
    [-0.01391844353092,    0.25694120850288,   -0.48716076856953],
    [-0.09144615671575,    0.02888016349900,    0.11803292616128],
    [ 0.20224696512519,   -0.21949067659852,   -0.03036909292113],
    [ 0.01391844353093,    0.25694120850288,   -0.48716076856953],
    [-0.09144615671576,   -0.02888016349901,   -0.11803292616128],
    [ 0.20224696512519,    0.21949067659851,    0.03036909292113],
    [-0.01391908738231,    0.25694995499878,    0.48715437825960],
    [-0.09144751666578,    0.02888116767566,   -0.11803119396265],
    [-0.20224409388994,    0.21948949440290,   -0.03037357515117],
    [ 0.01391908738231,    0.25694995499878,    0.48715437825960],
    [-0.09144751666578,   -0.02888116767567,    0.11803119396264],
    [-0.20224409388995,   -0.21948949440290,    0.03037357515118]
    ])

    print("\nError in Numerical vs. Analytic Tensors (a.u.):")
    print(analytic_ref-I)

    assert(np.max(np.abs(analytic_ref-I)) < 1e-6)
