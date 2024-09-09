# Introduction
'''
Functions to perform quantum circuit generation for HHL algorithm,
transpiling the circuit for the specific backend, and
running exact and shots-based simulations.
'''

import numpy as np
from linear_solvers import NumPyLinearSolver, HHL
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.execute_function import execute
from qiskit import Aer
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity

import time
import os
import argparse
import pickle

# get backend based on type and method
def qc_backend(backend_type, backend_method, args):
    if backend_type=='ideal':
        # ideal simulator
        backend = AerSimulator(method=backend_method)
    elif backend_type=='fake':
        # fake backend
        from qiskit.providers import fake_provider
        backend = getattr(fake_provider, backend_method)() # FakeNairobi FakePerth  FakeMumbai  FakeWashington
    elif backend_type=='real-ibm':
        # real hardware backend
        from qiskit_ibm_provider import IBMProvider
        from qiskit_ibm_runtime import QiskitRuntimeService
        # save your IBMProvider accout for future loading
        API_KEY = os.getenv('IBMQ_API_KEY')
        instance = os.getenv('IBMQ_INSTANCE')
        IBMProvider.save_account(instance=instance, token=API_KEY, overwrite=True)
        # save your QiskitRuntimeService accout for future loading
        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            instance=instance,
            token=API_KEY,
            overwrite=True
        )
        provider = IBMProvider()  # Using IBMProvider to use backend.run() option
        backend = provider.get_backend(backend_method)  # ibm_nairobi  simulator_statevector
    else:
        raise Exception(f'Backend type \'{backend_type}\' not implemented.')
    return backend

# circuit generation, transpile, running
def qc_circ(matrix, vector, hhl, args, backend_method, backend, classical_solution, filename='temp', plot_hist=False):

    MATRIX_SIZE = matrix.shape[0]
    NUM_QUBITS = int(np.log2(MATRIX_SIZE))
    print(f'**************************Quantum circuit generation, transpile & running*************************', flush=True)
    # ============================
    # 1. Generate circuit
    savefilename = f'{filename}_circ_nq{NUM_QUBITS}.pkl'
    if args.loadcirc == True:
        t = time.time()
        file = open(savefilename, 'rb')
        data = pickle.load(file)
        file.close()
        circ = data['circ']
        t_load = time.time() - t
        print(f'===============Loaded circuit (before transpile) using pickled data==============')
        print(f'Time elapsed for loading circuit:  {int(t_load/60)} min {t_load%60:.2f} sec', flush=True)
    else:
        print(f'==================Making a circuit and simulating it================', flush=True)
        t = time.time()
        circ = hhl.construct_circuit(matrix, vector)
        t_circ = time.time() - t
        print(f'Time elapsed for generating HHL circuit:  {int(t_circ/60)} min {t_circ%60:.2f} sec')
        # Save data
        if args.savedata == True:
            save_data = {   'NUM_QUBITS'            : NUM_QUBITS,
                            'matrix'                : matrix,
                            'vector'                : vector,                 
                            'circ'                  : circ,
                            't_circ'                : t_circ}
            file = open(savefilename, 'wb')
            pickle.dump(save_data, file)
            file.close()
            print("===========Circuit saved===========")
    print(f'Circuit:\n{circ}', flush=True)
    if backend_method[0]=='ideal': circ.save_statevector()
    circ.measure_all()
    
    # ============================
    # 2. Transpile circuit for simulator
    savefilename = f'{filename}_circ-transpile_nq{NUM_QUBITS}_backend-{backend_method[1]}.pkl'
    if args.loadcirctranspile == True:
        t = time.time()
        file = open(savefilename, 'rb')
        data = pickle.load(file)
        file.close()
        circ = data['circ']
        t_load = time.time() - t
        print(f'===============Loaded transpiled circuit using pickled data==============')
        print(f'Time elapsed for loading circuit:  {int(t_load/60)} min {t_load%60:.2f} sec', flush=True)
    else:
        t = time.time()
        circ = transpile(circ, backend)
        t_transpile = time.time() - t
        print(f'Time elapsed for transpiling the circuit:  {int(t_transpile/60)} min {t_transpile%60:.2f} sec')   

        # Save data
        if args.savedata == True:
            save_data = {   'NUM_QUBITS'            : NUM_QUBITS,
                            'matrix'                : matrix,
                            'vector'                : vector,                 
                            'circ'                  : circ,
                            't_circ'                : t_circ,
                            't_transpile'           : t_transpile}
            file = open(savefilename, 'wb')
            pickle.dump(save_data, file)
            file.close()
            print("===========Transpiled Circuit saved===========", flush=True)
    
    # ============================
    # 3. Run and get counts
    shots = args.SHOTS
    t = time.time()
    result = backend.run(circ, shots=shots, memory=True).result()
    # extracting the counts for the given number of counts based on the probabilities obtained from the true simulation
    t_run = time.time() - t
    print(f'Time elapsed for running the circuit:  {int(t_run/60)} min {t_run%60:.2f} sec', flush=True)   
    counts = result.get_counts(circ)   
    print(f'counts:\n{counts}')
    
    # Returning measurement outcomes for each shot
    memory = result.get_memory(circ)
    
    # Saving the final statevector if using ideal (qiskit) backend
    if backend_method[0]=='ideal':
        statevector = result.get_statevector()
        statevector = np.asarray(statevector)
        istart = int(len(statevector)/2)
        exact_vector = statevector[istart:istart+MATRIX_SIZE].real

    # get counts based probabilistic/statistical state vector
    counts_ancilla, counts_total, probs_vector, counts_vector = get_ancillaqubit(counts, NUM_QUBITS)
    print(f'All counts of ancila (only the first 2**nq represent solution vector):\n{counts_ancilla}')
    print("Counts vector should approach exact vector in infinite limit")
    print(f'counts_vector:\n{counts_vector}')
    if backend_method[0]=='ideal': print(f'exact_vector/norm:\n{exact_vector/np.linalg.norm(exact_vector)}')
    
    # print solutions
    print(f'\ntrue solution:\n{classical_solution.state}')
    # normalize counts vector with true solution norm
    counts_solution_vector = classical_solution.euclidean_norm * counts_vector / np.linalg.norm(counts_vector)
    print(f'\ncounts solution vector:\n{counts_solution_vector}')
    print(f'diff with true solution (%):\n{np.abs(classical_solution.state-counts_solution_vector)*100/(classical_solution.state+1e-15)}')
    print(f'Fidelity: {fidelity(counts_solution_vector, classical_solution.state)}')
    if backend_method[0]=='ideal':
        exact_solution_vector = classical_solution.euclidean_norm * exact_vector / np.linalg.norm(exact_vector)
        print(f'\nexact solution vector:\n{exact_solution_vector}')
        print(f'diff with true solution (%):\n{np.abs(classical_solution.state-exact_solution_vector)*100/(classical_solution.state+1e-15)}')
        print(f'Fidelity: {fidelity(exact_solution_vector, classical_solution.state)}')

    # plot histogram
    if plot_hist:
        from qiskit.tools.visualization import plot_histogram
        import matplotlib.pyplot as plt
        plot_histogram(counts, figsize=(7, 7), color='tab:green', title=f'{backend_method[0]}:{backend_method[1]}')  # dodgerblue tab:green
        plt.savefig('Figs/temp_hist.png')

    # Save full data
    savefilename = f'{filename}_circ-fullresults_nq{NUM_QUBITS}_backend-{backend_method[1]}_shots{shots}.pkl'
    if args.savedata == True:
        save_data = {   'NUM_QUBITS'                : NUM_QUBITS,
                        'matrix'                    : matrix,
                        'vector'                    : vector,                 
                        'circ'                      : circ,
                        'shots'                     : shots,
                        'result'                    : result,
                        'counts'                    : counts,
                        'memory'                    : memory,
                        'exact_vector'              : exact_vector,
                        'counts_ancilla'            : counts_ancilla,
                        'counts_vector'             : counts_vector,
                        'counts_solution_vector'    : counts_solution_vector,
                        'exact_solution_vector'     : exact_solution_vector,
                        'classical_solution'        : classical_solution,
                        't_circ'                    : t_circ,
                        't_transpile'               : t_transpile,
                        't_run'                     : t_run}
        file = open(savefilename, 'wb')
        pickle.dump(save_data, file)
        file.close()
        print("===========Full data saved===========")

# function to measure the qubits
def get_ancillaqubit(counts, nq):
    '''
    NOTE: only count measurements when ancilla qubit (leftmost) is 1
    input: 
        counts   counts from the simulator
        nq       number of qubits used to represent the system or solution vector
    output: 
        counts_ancill     acounts of the measurements where ancilla qubit = 1
        other metricis for combination of nq qubits = 1
    '''
    counts_list = list(counts.items())
    counts_ancilla = []
    ancilla_states = []
    # check the left most qubit
    for i in range(len(counts_list)):
        if counts_list[i][0][0]=='1': # extract all ancilla qubits=1
            counts_ancilla += (counts_list[i],)
            ancilla_states += (counts_list[i][0],)
    # sort based on right most qubits. Find the argsort and manually rearrange the counts list.
    ancilla_states_sortedID = np.argsort(ancilla_states)
    counts_ancilla_sorted = []
    for i in range(len(counts_ancilla)):
        counts_ancilla_sorted += (counts_ancilla[ancilla_states_sortedID[i]],)
    counts_ancilla = counts_ancilla_sorted.copy()
    # At this point, all the states are sorted such that ancilla=1 and the combination of nb qubits is 0 or 1
    # So, we take the first 2**nb states (OR size of the system)
    num_state = 2**nq
    # re-compute counts_total
    counts_total = 0
    for i in range(num_state):
        counts_total += counts_ancilla[i][1]
    # compute solution vectors
    probs_vector = []
    counts_vector = []
    for i in range(num_state):
        probs_vector += (counts_ancilla[i][1]/(1.0*counts_total),)
        counts_vector += (np.sqrt(counts_ancilla[i][1]/(1.0*counts_total)),)
    return counts_ancilla, counts_total, np.array(probs_vector), np.array(counts_vector)

# function to compute fidelity of the solution
def fidelity(qfunc, true):
    solution_qfun_normed = qfunc / np.linalg.norm(qfunc)
    solution_true_normed = true / np.linalg.norm(true)
    fidelity = state_fidelity(solution_qfun_normed, solution_true_normed)
    return fidelity


