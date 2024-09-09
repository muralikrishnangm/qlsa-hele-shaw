# # Introduction
'''
Script for developmental work on error modeling and mitigation.
Sample code run script:
python linear_solver_errormitigation.py -nq 1 -s 1000
NOTE: This code may require porting to latest version of Qiskit as it uses IBM's real/simulator backends.
'''

import numpy as np
from scipy.sparse import diags

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.execute_function import execute
from qiskit import Aer
from qiskit_aer import AerSimulator
from linear_solvers import NumPyLinearSolver, HHL
# library to generate HHL circuit for given matrix and vector, transpile circuit with given backend, and run shot-based  simulations
from func_qc_errormitigation import qc_backend, qc_circ

import time
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-nq", "--NUM_QUBITS", type=int, default=2, required=True, help="Numer of qubits to determine size of linear system of quations being solved. A matrix size = 2^NUM_QUBITS.")
parser.add_argument("-s", "--SHOTS", type=int, default=1000, required=True, help="Numer of shots for the simulator.")
parser.add_argument("-backmet", "--backend_method",  type=str, default='simulator_statevector', required=False, help="Method/name of the IBM backend. E.g. 'simulator_statevector' 'ibmq_qasm_simulator'")
parser.add_argument("--addnoise", default=False, action='store_true', help="Run a noisy ideal simulation, using fake backend to model noise.")
parser.add_argument("-noisebackmet", "--noise_backend_method",  type=str, default='FakeNairobi', required=False, help="Method/name of the fake/real backend to model noise. E.g. 'FakeMumbai' 'ibmq_mumbai' 'FakeNairobi' 'ibm_nairobi'")
parser.add_argument("-opt", "--opt_level", type=int, default=0, help="How much optimization to perform on the circuits. Higher levels generate more optimized circuits, at the expense of longer transpilation time.")
parser.add_argument("-dd", "--dynamical_decoupling", default=False, action='store_true', help="Enable dynamical decoupling.")
parser.add_argument("-res", "--res_level", type=int, default=0, help="Error mitigation strategy. 0: None; 1: TREX (minimal mitigation - readout errors); 2: ZNE (midium mitigation - bias in estimators); 3: PEC (heavy mitigation - zero estimator bias)")
parser.add_argument("--savedata", default=False, action='store_true', help="Save data at `models/<filename>` with `<filename>` based on parameters.")
parser.add_argument("--loadcirc", default=False, action='store_true', help="Load circuit at `models/<filename>` with `<filename>` based on parameters.")
parser.add_argument("--loadcirctranspile", default=False, action='store_true', help="Load transpiled circuit at `models/<filename>` with `<filename>` based on parameters.")
args = parser.parse_args()

# custom systems
NUM_QUBITS = args.NUM_QUBITS 
MATRIX_SIZE = 2 ** NUM_QUBITS

# save & load filename
savedata = args.savedata
savedir = f'models'
filename = f'{savedir}/sample_HHL_errormitigation'

# entries of the tridiagonal Toeplitz symmetric matrix
a = 1
b = -1/3
matrix = diags([b, a, b],
               [-1, 0, 1],
               shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
vector = np.array([1] + [0]*(MATRIX_SIZE - 1))

# setup HHL solver
backend_method = args.backend_method
print(f'Using ibmq simulator with \'{backend_method}\' backend')
service, backend = qc_backend(backend_method, args)
print(f'Backend: {backend}')
hhl = HHL(quantum_instance=backend)

# Solutions
# classical soultion
t = time.time()
classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))
t_classical = time.time() - t
print(f'Time elapsed for classical:  {int(t_classical/60)} min {t_classical%60:.2f} sec', flush=True)

# quantum circuit solution
qc_circ(matrix, vector, hhl, args, service, backend_method, backend, classical_solution, filename=filename)



