# Introduction
'''
Functions to perform error modeling and mitigation studies.
qiskit tutorial for noisy simulator and noise mitigation run: https://qiskit.org/ecosystem/ibm-runtime/how_to/noisy_simulators.html
NOTE: This code may require porting to latest version of Qiskit as it uses IBM's real/simulator backends.
'''

import numpy as np
from linear_solvers import NumPyLinearSolver, HHL
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.execute_function import execute
from qiskit import Aer
from qiskit_aer import AerSimulator
from qiskit.quantum_info import state_fidelity
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options

import time
import os
import argparse
import pickle

# get backend based on method
def qc_backend(backend_method, args):
    # real hardware backend
    # save your IBMProvider accout for future loading
    API_KEY = os.getenv('IBMQ_API_KEY')
    instance = os.getenv('IBMQ_INSTANCE')
    # save your QiskitRuntimeService accout for future loading
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        instance=instance,
        token=API_KEY,
        overwrite=True
    )
    service = QiskitRuntimeService()
    backend = service.get_backend(backend_method)  # ibm_nairobi  simulator_statevector ibmq_qasm_simulator
    return service, backend 

# circuit generation, transpile, running
def qc_circ(matrix, vector, hhl, args, service, backend_method, backend, classical_solution, filename='temp', plot_hist=False):

    MATRIX_SIZE = matrix.shape[0]
    NUM_QUBITS = int(np.log2(MATRIX_SIZE))
    print(f'**************************Quantum circuit generation, transpile & running*************************', flush=True)
    # ============================
    # 1. Generate circuit
    savefilename = f'{filename}_circ_nq{NUM_QUBITS}_backend-{backend_method}.pkl'
    if args.loadcirc == True:
        t = time.time()
        file = open(savefilename, 'rb')
        data = pickle.load(file)
        file.close()
        circ = data['circ']
        t_circ = time.time() - t
        print(f'===============Loaded circuit (before transpile) using pickled data==============')
        print(f'Time elapsed for loading circuit:  {int(t_circ/60)} min {t_circ%60:.2f} sec', flush=True)
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
                            't_circ'                : t_circ,
                            'args'                  : args}
            file = open(savefilename, 'wb')
            pickle.dump(save_data, file)
            file.close()
            print("===========Circuit saved===========")
            # print(circ.qasm()) #filename=f'{savefilename}_qasm')
    print(f'Circuit:\n{circ}', flush=True)
    circ.measure_all()
   
    # ============================
    # 2. Transpile circuit for simulator
    savefilename = f'{filename}_circ-transpile_nq{NUM_QUBITS}_backend-{backend_method}.pkl'
    if args.loadcirctranspile == True:
        t = time.time()
        file = open(savefilename, 'rb')
        data = pickle.load(file)
        file.close()
        circ = data['circ']
        t_transpile = time.time() - t
        print(f'===============Loaded transpiled circuit using pickled data==============')
        print(f'Time elapsed for loading circuit:  {int(t_transpile/60)} min {t_transpile%60:.2f} sec', flush=True)
    else:
        t = time.time()
        circ = transpile(circ, backend, optimization_level=args.opt_level, seed_transpiler=2024)
        # dynamical decoupling
        if args.dynamical_decoupling:
          if "simulator" in backend_method:
            raise Exception(f'Dynamical decoupling only works for real backend, not for ideal.')
          print(f'Dynamical decoupling enabled.')
          from qiskit.transpiler import PassManager
          from qiskit.transpiler.passes import ASAPScheduleAnalysis, PadDynamicalDecoupling
          from qiskit.circuit.library import XGate
          # Get gate durations so the transpiler knows how long each operation takes
          durations = backend.target.durations()
          # This is the sequence we'll apply to idling qubits
          dd_sequence = [XGate(), XGate()]
          # Run scheduling and dynamic decoupling passes on circuit
          pm = PassManager([ASAPScheduleAnalysis(durations), PadDynamicalDecoupling(durations, dd_sequence, pulse_alignment=backend.configuration().timing_constraints['pulse_alignment'])]) # default pulse_alignment=1 which is not true for various backends
          circ = pm.run(circ)
        t_transpile = time.time() - t
        print(f'Time elapsed for transpiling the circuit:  {int(t_transpile/60)} min {t_transpile%60:.2f} sec')   

        if args.savedata == True:
            save_data = {   'NUM_QUBITS'            : NUM_QUBITS,
                            'matrix'                : matrix,
                            'vector'                : vector,                 
                            'circ'                  : circ,
                            't_circ'                : t_circ,
                            't_transpile'           : t_transpile,
                            'args'                  : args}
            file = open(savefilename, 'wb')
            pickle.dump(save_data, file)
            file.close()
            print("===========Transpiled Circuit saved===========", flush=True)
    
    # ============================
    # 3. Run IBM simulator and get probability dist
    shots = args.SHOTS
    options = Options()
    if args.addnoise: 
        if "Fake" in args.noise_backend_method: # fake backend to model noise
            from qiskit.providers import fake_provider
            backend_noise = getattr(fake_provider, args.noise_backend_method)() # FakeNairobi FakePerth  FakeMumbai  FakeWashington
            print(f'Using fake backend for noise model: \'{backend_noise}\'')
        else: # Get noise model from real backend
            backend_noise = service.get_backend(args.noise_backend_method)  # ibm_nairobi  simulator_statevector ibmq_qasm_simulator
            print(f'Using real backend for noise model: \'{backend_noise}\'')
        # Get noise model from backend_noise
        from qiskit.providers.aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(backend_noise)
        # Get coupling map from backend_noise
        coupling_map = backend_noise.configuration().coupling_map
        # Get basis gates from noise model
        basis_gates = noise_model.basis_gates
        # Set options
        options.simulator = {
        "noise_model": noise_model,
        "basis_gates": basis_gates,
        "coupling_map": coupling_map
        }
        print(f'Noise model: {noise_model}')
    # Set optimization_level. Not relevant if skip_transpilation
    options.optimization_level = args.opt_level # 0: no optimization
    print(f'Optimizing with level: {args.opt_level}')
    # Set error mitigation level
    options.resilience_level = args.res_level # 0: no error mitigation
    print(f'Mitigating noise with level: {args.res_level}')
 
    # options.simulator.seed_simulator = 2024
    options.transpilation.skip_transpilation = True
    options.execution.shots = shots
    
    t = time.time()
    with Session(service=service, backend=backend):
        sampler = Sampler(options=options)
        job = sampler.run(circ)
        result = job.result()
    t_run = time.time() - t
    print(f'Time elapsed for running the circuit:  {int(t_run/60)} min {t_run%60:.2f} sec', flush=True)
    # print(f">>> Quasi-probability distribution: {result.quasi_dists}")
    print(f">>> Quasi-probability distribution: {result.quasi_dists[0].binary_probabilities()}") # same format as counts

    counts = result.quasi_dists[0].binary_probabilities()
    print(f'counts:\n{counts}')
    
    # get counts based probabilistic/statistical state vector
    counts_ancilla, counts_total, probs_vector, counts_vector = get_ancillaqubit(counts, NUM_QUBITS)
    print(f'All counts of ancila (only the first 2**nq represent solution vector):\n{counts_ancilla}')
    print("Counts vector should approach exact vector in infinite limit")
    print(f'counts_vector:\n{counts_vector}')
    # counts vector is already normalized
    counts_solution_vector = classical_solution.euclidean_norm * counts_vector / np.linalg.norm(counts_vector)
    print(f'true solution:\n{classical_solution.state}')
    print(f'counts solution vector:\n{counts_solution_vector}')
    print(f'diff with true solution (%):\n{np.abs(classical_solution.state-counts_solution_vector)*100/(classical_solution.state+1e-15)}')
    print(f'Fidelity: {fidelity(counts_solution_vector, classical_solution.state)}')

    # Save full data
    savefilename = f'{filename}_circ-fullresults_nq{NUM_QUBITS}_backend-{backend_method}_shots{shots}_addnoise{int(args.addnoise)}_mitg{int(args.opt_level>0 or args.res_level>0)}.pkl'
    if args.savedata == True:
        save_data = {   'NUM_QUBITS'                : NUM_QUBITS,
                        'matrix'                    : matrix,
                        'vector'                    : vector,                 
                        'circ'                      : circ,
                        'shots'                     : shots,
                        'result'                    : result,
                        'counts'                    : counts,
                        'counts_ancilla'            : counts_ancilla,
                        'counts_vector'             : counts_vector,
                        'counts_solution_vector'    : counts_solution_vector,
                        'classical_solution'        : classical_solution,
                        't_circ'                    : t_circ,
                        't_transpile'               : t_transpile,
                        't_run'                     : t_run,
                        'args'                      : args}
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


