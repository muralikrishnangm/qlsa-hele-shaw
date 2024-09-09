# Quantum Linear Solver Algorithm

A sample implementation of solving linear system of equations using [Qiskit's HHL implementation](https://learn.qiskit.org/course/ch-applications/solving-linear-systems-of-equations-using-hhl-and-its-qiskit-implementation). The fluid flow use-cases are following the work of [Bharadwaj & Srinivasan (2020)](https://www.sto.nato.int/publications/STO%20Educational%20Notes/STO-EN-AVT-377/EN-AVT-377-01.pdf). The implementations are using python scripts and Jupyter notebooks which use Quiskit libraries. See [Qiskit](https://qiskit.org/documentation/getting_started.html) it get started with installation to local conda environments. Or run on [IBM quantum-computing](https://quantum-computing.ibm.com/) platform.

* Contributors:
      * Murali Gopalakrishnan Meena (Oak Ridge National Laboratory)
      * Kalyan Gottiparthi (Oak Ridge National Laboratory)
      * Justin Lietz (NVIDIA)

# Installation

1. [Optional] Steps for your own mini-conda installation
      ```
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      bash Miniconda3-latest-Linux-x86_64.sh -b -p /path/to/your/local/dir/for/miniconda
      source /path/to/your/local/dir/for/miniconda/bin/activate
      ```
2. Make custom conda env
      ```
      conda create --name env-qlsa python=3.11
      conda activate env-qlsa
      ```
3. Install qiskit
      ```
      pip install -r requirements.txt --no-cache-dir
      ```

4. Test qiskit installation: [`test_qiskit_installation.py`](test_qiskit_installation.py)
      ```
      python test_qiskit_installation.py -backtyp ideal
      ```  
      <details><summary>Click here for sample output from the test code</summary>

      ```
      Backend: qasm_simulator
      Job status is JobStatus.DONE

      Total count for 00 and 11 are: {'11': 476, '00': 524}
           ┌───┐     ┌─┐   
      q_0: ┤ H ├──■──┤M├───
           └───┘┌─┴─┐└╥┘┌─┐
      q_1: ─────┤ X ├─╫─┤M├
                └───┘ ║ └╥┘
      c: 2/═══════════╩══╩═
                      0  1 
      ```

      </details>

      * Change `-backtyp` for different backends.
      * To run using IBM Provider, you need to add your IBM Quantum Computing API KEY and instance to the `keys.sh` file and source activate it. **NOTE:** You may have to port your codes to the latest Qiskit version for this.
5. Test [linear solver package](https://github.com/anedumla/quantum_linear_solvers): [`test_linear_solver.py`](test_linear_solver.py)

      ```
      python test_linear_solver.py -nq 2
      ```

      <details><summary>Click here for sample output from the test code:</summary>

      ```
      Simulator: aer_simulator_statevector
      Time elapsed for classical:  0 min 0.00 sec
      Time elapsed for naive HHL:  0 min 0.53 sec
      Time elapsed for tridi-Toep: 0 min 1.68 sec
      classical state: [1.14545455 0.43636364 0.16363636 0.05454545]
      naive state:
            ┌──────────────┐┌──────┐        ┌─────────┐
      q9_0: ┤0             ├┤4     ├────────┤4        ├
            │  circuit-165 ││      │        │         │
      q9_1: ┤1             ├┤5     ├────────┤5        ├
            └──────────────┘│      │┌──────┐│         │
      q10_0: ───────────────┤0     ├┤3     ├┤0        ├
                            │  QPE ││      ││  QPE_dg │
      q10_1: ───────────────┤1     ├┤2     ├┤1        ├
                            │      ││      ││         │
      q10_2: ───────────────┤2     ├┤1 1/x ├┤2        ├
                            │      ││      ││         │
      q10_3: ───────────────┤3     ├┤0     ├┤3        ├
                            └──────┘│      │└─────────┘
      q11: ───────────────-─────────┤4     ├───────────
                                    └──────┘           
      tridiagonal state:
            ┌──────────────┐┌──────┐        ┌─────────┐
      q82_0:┤0             ├┤4     ├────────┤4        ├
            │  circuit-521 ││      │        │         │
      q82_1:┤1             ├┤5     ├────────┤5        ├
            └──────────────┘│      │┌──────┐│         │
      q83_0:────────────────┤0     ├┤3     ├┤0        ├
                            │      ││      ││         │
      q83_1:────────────────┤1 QPE ├┤2     ├┤1 QPE_dg ├
                            │      ││      ││         │
      q83_2:────────────────┤2     ├┤1     ├┤2        ├
                            │      ││  1/x ││         │
      q83_3:────────────────┤3     ├┤0     ├┤3        ├
                            │      ││      ││         │
      a1:   ────────────────┤6     ├┤      ├┤6        ├
                            └──────┘│      │└─────────┘
      q84:  ────────────────────────┤4     ├───────────
                                    └──────┘           
      classical Euclidean norm:    1.237833351044751
      naive Euclidean norm:        1.2099806231118977 (diff (%): 2.250e+00)
      tridiagonal Euclidean norm:  1.2099204004859732 (diff (%): 2.255e+00)
      classical state:
      [1.14545455 0.43636364 0.16363636 0.05454545]
      full solution vector (naive):
      # of elements in solution vector: 128
      [1.11266151 0.43866345 0.16004585 0.08942688]
      diff (%): [ 2.86288363  0.52703993  2.1942013  63.94928497]
      full solution vector (tridi):
      # of elements in solution vector: 256
      [1.11261363 0.4386119  0.16005021 0.08945291]
      diff (%): [ 2.8670642   0.5152269   2.19153945 63.9969963 ]
      ===========Data not saved===========
      ```
      </details>
      
      * Change `-nq` to change size of system of equations.

# Run

1. Load Python environemnt:
      ```
      source /path/to/your/local/dir/for/miniconda/bin/activate
      conda activate env-qlsa
      ```
2. Run generalized QLSA script for various use-cases: [`linear_solver.py`](linear_solver.py)
    ```
    python linear_solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000
    ```
    * See [`input_vars.yaml`](input_vars.yaml) and [`func_matrix_vector.py`](func_matrix_vector.py) for various cases available.

      <details><summary>Click here for sample output for Hele-Shaw flow, solving for pressure:</summary>

      ```
      Case: Hele-Shaw
      Solving analytically...
      Solving numerically...
      =====Solving for pressure...=====
      Using analytical pressure profile...
      Determinant of resulting matrix: 1.0
      Condition # of resulting matrix: 1.0
      Determinant of resulting matrix: 1.0
      Size of A & B are not power of 2:
      Next 2 power of 6 = 8
      Value to be padded = 2
      Padded shape of A: (6, 6) -> (8, 8)
      Padded shape of B: (6,) -> (8,)
      Padded A with diag 1:
      [[1. 0. 0. 0. 0. 0. 0. 0.]
      [0. 1. 0. 0. 0. 0. 0. 0.]
      [0. 0. 1. 0. 0. 0. 0. 0.]
      [0. 0. 0. 1. 0. 0. 0. 0.]
      [0. 0. 0. 0. 1. 0. 0. 0.]
      [0. 0. 0. 0. 0. 1. 0. 0.]
      [0. 0. 0. 0. 0. 0. 1. 0.]
      [0. 0. 0. 0. 0. 0. 0. 1.]]
      Padded B:
      [200.   0. 200.   0. 200.   0.   0.   0.]
      Determinant of resulting matrix: 1.0
      Reformatted A_herm:
      [[1. 0. 0. 0. 0. 0. 0. 0.]
      [0. 1. 0. 0. 0. 0. 0. 0.]
      [0. 0. 1. 0. 0. 0. 0. 0.]
      [0. 0. 0. 1. 0. 0. 0. 0.]
      [0. 0. 0. 0. 1. 0. 0. 0.]
      [0. 0. 0. 0. 0. 1. 0. 0.]
      [0. 0. 0. 0. 0. 0. 1. 0.]
      [0. 0. 0. 0. 0. 0. 0. 1.]]
      B_herm:
      [217.32050808  17.32050808 217.32050808  17.32050808 217.32050808
      17.32050808  17.32050808  17.32050808]
      Determinant of resulting matrix: 1.0
      Condition # of resulting matrix: 1.0
      Using 'ideal' simulator with 'statevector' backend
      Backend: AerSimulator('aer_simulator')
      Time elapsed for classical:  0 min 0.00 sec
      **************************Quantum circuit generation, transpile & running*************************
      ==================Making a circuit and simulating it================
      Time elapsed for generating HHL circuit:  0 min 1.42 sec
      Circuit:
            ┌──────────────┐┌──────┐        ┌─────────┐
      q9_0: ┤0             ├┤5     ├────────┤5        ├
            │              ││      │        │         │
      q9_1: ┤1 circuit-165 ├┤6     ├────────┤6        ├
            │              ││      │        │         │
      q9_2: ┤2             ├┤7     ├────────┤7        ├
            └──────────────┘│      │┌──────┐│         │
      q10_0:────────────────┤0     ├┤4     ├┤0        ├
                            │  QPE ││      ││  QPE_dg │
      q10_1:────────────────┤1     ├┤3     ├┤1        ├
                            │      ││      ││         │
      q10_2:────────────────┤2     ├┤2     ├┤2        ├
                            │      ││  1/x ││         │
      q10_3:────────────────┤3     ├┤1     ├┤3        ├
                            │      ││      ││         │
      q10_4:────────────────┤4     ├┤0     ├┤4        ├
                            └──────┘│      │└─────────┘
      q11:  ────────────────────────┤5     ├───────────
                                    └──────┘           
      Time elapsed for transpiling the circuit:  0 min 0.16 sec
      Time elapsed for running the circuit:  0 min 0.04 sec
      counts:
      {'100000101': 9, '100000011': 6, '100000110': 10, '100000010': 1682, '100000111': 15, '100000100': 1645, '100000001': 7, '100000000': 1626}
      All counts of ancila (only the first 2**nq represent solution vector):
      [('100000000', 1626), ('100000001', 7), ('100000010', 1682), ('100000011', 6), ('100000100', 1645), ('100000101', 9), ('100000110', 10), ('100000111', 15)]
      Counts vector should approach exact vector in infinite limit
      counts_vector:
      [0.5702631  0.03741657 0.58       0.03464102 0.57358522 0.04242641
      0.04472136 0.05477226]
      exact_vector/norm:
      [0.57431815 0.04577332 0.57431815 0.04577332 0.57431815 0.04577332
      0.04577332 0.04577332]

      true solution:
      [0.57431815 0.04577332 0.57431815 0.04577332 0.57431815 0.04577332
      0.04577332 0.04577332]

      counts solution vector:
      [0.5702631  0.03741657 0.58       0.03464102 0.57358522 0.04242641
      0.04472136 0.05477226]
      diff with true solution (%):
      [ 0.70606392 18.25681403  0.98932091 24.32051541  0.12761819  7.31193938
      2.29820551 19.65977172]
      Fidelity: 0.9996637113875608

      exact solution vector:
      [0.57431815 0.04577332 0.57431815 0.04577332 0.57431815 0.04577332
      0.04577332 0.04577332]
      diff with true solution (%):
      [1.93311499e-13 6.74586741e-12 3.28629548e-13 4.07783895e-12
      5.79934496e-14 6.59427488e-12 3.31987632e-12 6.82166367e-13]
      Fidelity: 1.0
      ```
      </details>
2. Run error modeling and mitigation studies: [`linear_solver_errormitigation.py`](linear_solver_errormitigation.py)
    ```
    python linear_solver_errormitigation.py -nq 1 -s 1000
    ```
    * **NOTE**: Please note that you may have to port the code to the latest Qiskit version as this script is meant to use IBM's hardware/simulators.

# Jupyter notebooks for visualizing results

* [plot_Hele-Shaw.ipynb](plot_Hele-Shaw.ipynb): visualizing the implementation of Qiskit's HHL for solving 2D Hele-Shaw flow problem.
* [plot_errormitigation.ipynb](plot_errormitigation.ipynb): visualizing the results of the error modeling and mitigation/suppression studies.
* [plot_metrics.ipynb](plot_metrics.ipynb): visualizing the computational cost and other metrics with varying system size.

## Using custom env on JupyterLab

* Resources:
  * [OLCF JupyterHub](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#jupyter-at-olcf)
  * [NERSC JupyterHub](https://docs.nersc.gov/services/jupyter/)
* Always install and start JupyterLab in your base conda env.
* Use custom kernels as needed. See below for how to install custom kernels.
* To import your custom conda env to JupyterLab, follow the steps below which are modified from [OLCF Jupyter docs](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#example-creating-a-conda-environment-for-rapids):
  * Install JupyterLab in your custom conda env. Do the rest of the steps in your base env.
  * Follow steps 1-2 in [OLCF Jupyter docs](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#example-creating-a-conda-environment-for-rapids): Open a Terminal on JupyterLab using the Launcher.
  * Skip step 3: You don't have to create your own custom conda env as you have already done this.
  * Follow step 4 (source activate your custom env) using the custom env you created.
  * Follow step 5 (make your env visible in JupyterLab) using your desired env name: `python -m ipykernel install --user --name [env-name] --display-name [env-name]`. You may have to pip install the library `wcwidth` on the Jupyter terminal: `pip install wcwidth`
  * Finally refresh your page and the Lancher (and kernel selector for notebooks) will have your env.



