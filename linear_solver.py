# Introduction
'''
Script to perform quantum linear solver for any Ax=b problem. The HHL algorithm is used as the quantum linear solver algorithm.
Function `func_matrix_vector.py` is used to define A and b.
Function `func_qc.py` is used to generate, transpile, and run the quantum circuit.
Sample code run script:
python linear_solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000
'''

import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.execute_function import execute
from qiskit import Aer
from qiskit_aer import AerSimulator
from linear_solvers import NumPyLinearSolver, HHL
# library to generate HHL circuit for given matrix and vector, transpile circuit with given backend, and run shot-based  simulations
from func_qc import qc_backend, qc_circ
import func_matrix_vector as matvec

import time
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case_name",  type=str, default='ideal', required=False, help="Name of the problem case: 'sample-tridiag', 'hele-shaw'")
parser.add_argument("-casefile", "--case_variable_file",  type=str, default='ideal', required=False, help="YAML file containing variables for the case: 'input_vars.yaml'")

parser.add_argument("-s", "--SHOTS", type=int, default=1000, required=True, help="Numer of shots for the simulator.")
parser.add_argument("-backtyp", "--backend_type",  type=str, default='ideal', required=False, help="Type of the backend: 'ideal', 'fake' 'real-ibm'. NOTE: For `real-ibm`, you may need to port the code based on latest Qiskit version and implementations to run on hardware.")
parser.add_argument("-backmet", "--backend_method",  type=str, default='statevector', required=False, help="Method/name of the backend. E.g. 'statevector' 'FakeNairobi' 'ibm_nairobi'")

parser.add_argument("--savedata", default=False, action='store_true', help="Save data at `models/<filename>` with `<filename>` based on parameters.")
parser.add_argument("--loadcirc", default=False, action='store_true', help="Load circuit at `models/<filename>` with `<filename>` based on parameters.")
parser.add_argument("--loadcirctranspile", default=False, action='store_true', help="Load transpiled circuit at `models/<filename>` with `<filename>` based on parameters.")
args = parser.parse_args()

# Get system matrix and vector
matrix, vector, filename = matvec.get_matrix_vector(args)

# setup HHL solver
backend_init = qc_backend('ideal', 'statevector', args)
hhl = HHL(quantum_instance=backend_init)

# setup quantum backend
backend_type = args.backend_type
backend_method = args.backend_method
print(f'Using \'{backend_type}\' simulator with \'{backend_method}\' backend')
backend = qc_backend(backend_type, backend_method, args)
print(f'Backend: {backend}')

# Solutions
# classical soultion
t = time.time()
classical_solution = NumPyLinearSolver().solve(matrix, vector/np.linalg.norm(vector))
t_classical = time.time() - t
print(f'Time elapsed for classical:  {int(t_classical/60)} min {t_classical%60:.2f} sec', flush=True)

# quantum circuit solution
qc_circ(matrix, vector, hhl, args, [backend_type, backend_method], backend, classical_solution, filename=filename)



