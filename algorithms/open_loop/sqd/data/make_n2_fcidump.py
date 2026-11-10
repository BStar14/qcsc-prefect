"""Write the FCIDUMP of the N2 example: 6-31G at 1.1 Å, 1s cores frozen.

The active space has 16 spatial orbitals and (5, 5) electrons, so the LUCJ
circuit uses 32 qubits. The integrals are in the RHF MO basis, and the frozen
core energy plus the nuclear repulsion are written as the FCIDUMP constant.

Run it from the repository root to regenerate the file::

    python algorithms/open_loop/sqd/data/make_n2_fcidump.py
"""

from pathlib import Path

from pyscf import ao2mo, gto, mcscf, scf, tools

ATOM = "N 0 0 0; N 0 0 1.1"
BASIS = "6-31g"
N_FROZEN = 2
OUTPUT = Path(__file__).with_name("n2_631g_fc.fcidump")


def main() -> None:
    mol = gto.M(atom=ATOM, basis=BASIS, symmetry=False, verbose=0)
    mf = scf.RHF(mol).run()
    norb = mol.nao_nr() - N_FROZEN
    nelec = (mol.nelec[0] - N_FROZEN, mol.nelec[1] - N_FROZEN)

    cas = mcscf.CASCI(mf, ncas=norb, nelecas=nelec)
    h1, ecore = cas.get_h1eff()
    h2 = ao2mo.restore(1, cas.get_h2eff(), norb)
    tools.fcidump.from_integrals(str(OUTPUT), h1, h2, norb, sum(nelec), nuc=ecore, ms=0, tol=1e-15)
    print(f"Wrote {OUTPUT}: norb={norb}, nelec={nelec}, E(RHF)={mf.e_tot:.8f} Ha")


if __name__ == "__main__":
    main()
