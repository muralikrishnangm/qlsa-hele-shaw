import numpy as np
import time
import os
from qiskit import QuantumCircuit, transpile
from qiskit.providers.jobstatus import JobStatus

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-backtyp", "--backend_type",  type=str, default='ideal', required=False, help="Type of the backend: 'ideal', 'fake' 'real-ibm'")
args = parser.parse_args()

backend_type = args.backend_type
# Choose the simulator or backend to run the quantum circuit
if backend_type=='ideal':
  # Using ideal simulator, Aer's qasm_simulator (works even without IBMQ account, don't have to wait in a queue)
  from qiskit.providers.aer import QasmSimulator
  backend = QasmSimulator()
elif backend_type=='fake':
  # Using qiskit's fake provider (works even without IBMQ account, don't have to wait in a queue)
  from qiskit.providers import fake_provider
  backend = getattr(fake_provider, 'FakeNairobi')() #  FakePerth  FakeMumbai  FakeWashington
elif backend_type=='real-ibm':
  # Using the latest qiskit_ibm_provider
  #### IF YOU HAVE AN IBMQ ACCOUNT (using an actual backend) #####
  
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
  provider = IBMProvider()
  backend = provider.get_backend("simulator_statevector")  # ibm_nairobi  simulator_statevector
else:
    raise Exception(f'Backend type \'{backend_type}\' not implemented.')

print(f'Backend: {backend}')
######################################

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, backend)

# Execute the circuit on the qasm simulator
job = backend.run(compiled_circuit, shots=1000)

# Make a "waiting in queue" message
while job.status() is not JobStatus.DONE:
    print("Job status is", job.status() )
    time.sleep(30)

print("Job status is", job.status() )

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
print(circuit.draw())
