# # Introduction
# 
# Script to test the linear solver installation.
# Walkthrough of [HHL tutorial on Qiskit](https://learn.qiskit.org/course/ch-applications/solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation)
# for solving linear system of equations

import numpy as np
from scipy.sparse import diags
# Importing Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from linear_solvers import NumPyLinearSolver, HHL
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz

import time
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-nq", "--NUM_QUBITS", type=int, default=2, required=True, help="Numer of qubits to determine size of linear system of quations (A*x=b) being solved. Size of A matrix = 2^NUM_QUBITS.")
parser.add_argument("--savedata", default=False, action='store_true', help="Save data at `models/<filename>` with `<filename>` based on parameters.")
args = parser.parse_args()



# ============
# Generate matrix and vector
# We use a sample tridiagonal system. It's 2x2 version is:
#   matrix = np.array([ [1, -1/3], [-1/3, 1] ])
#   vector = np.array([1, 0])
#   tridi_matrix = TridiagonalToeplitz(1, 1, -1 / 3)
# custom systems
NUM_QUBITS = args.NUM_QUBITS 
MATRIX_SIZE = 2 ** NUM_QUBITS
# entries of the tridiagonal Toeplitz symmetric matrix
a = 1
b = -1/3
matrix = diags([b, a, b],
               [-1, 0, 1],
               shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
# we also generate an optimized matrix construction - tridiagonal toeplitz
tridi_matrix = TridiagonalToeplitz(NUM_QUBITS, a, b)

# ============
# Select backend: Using different simulators (default in `linear_solvers` is statevector simulation)
from qiskit import Aer
# https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html
# run `Aer.backends()` to see simulators
backend = Aer.get_backend('aer_simulator_statevector')
backend.set_options(precision='single')

# ============
# Setup HHL solver
hhl = HHL(1e-3, quantum_instance=backend)
print(f'Simulator: {backend}')

# ============
# Solutions
# Classical
t = time.time()
classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))
elpsdt = time.time() - t
print(f'Time elapsed for classical:  {int(elpsdt/60)} min {elpsdt%60:.2f} sec')
# HHL - normal matrix construction
t = time.time()
naive_hhl_solution = hhl.solve(matrix, vector)
elpsdt = time.time() - t
print(f'Time elapsed for naive HHL:  {int(elpsdt/60)} min {elpsdt%60:.2f} sec')
# HHL - tridiagonal toeplitz  matrix construction
t = time.time()
tridi_solution = hhl.solve(tridi_matrix, vector)
elpsdt = time.time() - t
print(f'Time elapsed for tridi-Toep: {int(elpsdt/60)} min {elpsdt%60:.2f} sec')

# ============
# Output states
# classical solution states
print('classical state:', classical_solution.state)
# quantum states - circuits
print('naive state:')
print(naive_hhl_solution.state)
print('tridiagonal state:')
print(tridi_solution.state)

# ============
# Comparing the observable - Euclidean norm
print(f'classical Euclidean norm:    {classical_solution.euclidean_norm}')
print(f'naive Euclidean norm:        {naive_hhl_solution.euclidean_norm} (diff (%): {np.abs(classical_solution.euclidean_norm-naive_hhl_solution.euclidean_norm)*100/classical_solution.euclidean_norm:1.3e})')
print(f'tridiagonal Euclidean norm:  {tridi_solution.euclidean_norm} (diff (%): {np.abs(classical_solution.euclidean_norm-tridi_solution.euclidean_norm)*100/classical_solution.euclidean_norm:1.3e})')

# ============
# Comparing the solution vectors component-wise
"""
https://learn.qiskit.org/course/ch-applications/solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation#implementationsim

Recall that the HHL algorithm can find a solution exponentially faster in the size of the system
than their classical counterparts (i.e. logarithmic complexity instead of polynomial). 
However the cost for this exponential speedup is that we do not obtain the full solution vector $x$. 
Instead, we obtain a quantum state representing the vector $x$ and learning all the components of 
this vector would take a linear time in its dimension, diminishing any speedup obtained by the quantum algorithm.
Therefore, we can only compute functions from $x$ (the so called observables) to learn information about the solution.

Comparing the solution vectors component-wise is more tricky, reflecting again the idea that we cannot obtain
the full solution vector from the quantum algorithm. However, for educational purposes we can check that indeed 
the different solution vectors obtained are a good approximation at the vector component level as well. 

To do so first we need to use Statevector from the quantum_info package and extract the right vector components, 
i.e. those corresponding to the ancillary qubit (bottom in the circuits) being 1 and 
the work qubits (the two middle in the circuits) being 0. Thus, we are interested in the states 10000 and 10001, 
corresponding to the first and second components of the solution vector respectively.
"""
from qiskit.quantum_info import Statevector
def get_solution_vector(solution, nstate):
    """Extracts and normalizes simulated state vector
    from LinearSolverResult."""
    # solution_vector = Statevector(solution.state).data[-nstate:].real
    temp = Statevector(solution.state)
    ID = np.where(np.abs(temp.data[:].real)<1e-10)[0]
    A = temp.data[:]
    A[ID] = 0+0j
    B = temp.data[:].real
    B[ID] = 0
    print(f'# of elements in solution vector: {len(B)}')
    istart = int(len(B)/2)
    solution_vector = temp.data[istart:istart+nstate].real
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)

print('classical state:')
print(f'{classical_solution.state}')
print('full solution vector (naive):')
solvec_naive = get_solution_vector(naive_hhl_solution, MATRIX_SIZE)
print(f'{solvec_naive}')
print(f'diff (%): {np.abs(classical_solution.state-solvec_naive)*100/classical_solution.state}')
print('full solution vector (tridi):')
solvec_tridi = get_solution_vector(tridi_solution, MATRIX_SIZE)
print(f'{solvec_tridi}')
print(f'diff (%): {np.abs(classical_solution.state-solvec_tridi)*100/classical_solution.state}')

# ============
# Save data
savedata = args.savedata
if savedata == True:
    savedir = f'models'
    savefilename = f'{savedir}/sample_HHL_numq{NUM_QUBITS}_fulldata'
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    n=2
    while os.path.exists(f'{savefilename}.npz'):  # avoid overwriting files
        savefilename = savefilename + f'v{n}' 
        n+=1

    savefilename += '.pkl' 
    save_data = {   'NUM_QUBITS'            : NUM_QUBITS,
                    'a'                     : a,
                    'b'                     : b,
                    'matrix'                : matrix,
                    'vector'                : vector,                 
                    'tridi_matrix'          : tridi_matrix,
                    'classical_solution'    : classical_solution,
                    'naive_hhl_solution'    : naive_hhl_solution,
                    'tridi_solution'        : tridi_solution,
                    'solvec_naive'          : solvec_naive,
                    'solvec_tridi'          : solvec_tridi,
                    'sim_solution'          : sim_solution}
    file = open(savefilename, 'wb')
    pickle.dump(save_data, file)
    file.close()

    print("===========Data saved===========")
else: 
    print("===========Data not saved===========")


