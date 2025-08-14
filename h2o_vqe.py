
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock 
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit_nature.second_q.drivers import MethodType
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)


import numpy as np
from scipy.optimize import minimize
import logging
logging.getLogger('qiskit').setLevel(logging.WARNING)

from stalk import ParameterStructure
from stalk.params import PesFunction

#building the driver for the molecule in question
def build_driver(atom, basis='ccpvdz', charge = 0, spin = 0):
    return PySCFDriver(
    atom=atom,
    basis=basis,
    charge=charge,
    spin=spin,
    method=MethodType.RHF,
    unit=DistanceUnit.ANGSTROM,
    )

# applying active space reductions to make the circuit size maneageable
def active_space(problem, active_orbitals = [1,2,3,4,5]):
    # gettign the number of electrons and orbitals to do the space transformation
    num_electrons = problem.num_alpha + problem.num_beta - active_orbitals[0]*2
    num_active_orbitals = len(active_orbitals)
    
    transformer = ActiveSpaceTransformer(
        num_electrons=num_electrons,
        num_spatial_orbitals=num_active_orbitals,
        active_orbitals = active_orbitals
    )
    problem = transformer.transform(problem)
    return problem

def build_ansatz(problem, mapper):
    return UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
        ),
    )

backend = AerSimulator()
def cost_func(theta, ansatz, estimator, observables, shots):
    if len(theta.shape) ==1:
	    theta = theta.reshape(1,-1)

    #preparing the circuit
    ref = QuantumCircuit(ansatz.num_qubits)
    circ = ref.compose(ansatz)

    pass_manager = generate_preset_pass_manager(3, backend)
    isa_circuit = pass_manager.run(circ)
    
    # running the crcuit on statevectorestimator
    pub = (isa_circuit,[observables], theta)

    if shots is None:
        accurate_energy = estimator.run([pub])
    else:
        accurate_energy = estimator.run([pub], precision = 1/np.sqrt(shots))

    energy = accurate_energy.result()[0].data.evs[0]
    # print("cost:", energy)
    return energy



def kernel_vqe(structure: ParameterStructure, 
               basis='ccpvdz', 
               charge=0, 
               spin=0, 
               active_orbitals=None, 
               trials =10, 
               optimizer = "L-BFGS-B", 
               shots = None, 
               p_dep = None, 
               **kwargs
               ):
   
    # Define the driver
    atom = [f"{el} {x} {y} {z}" for el, (x, y, z) in zip(structure.elem, structure.pos)]

    driver = build_driver(atom, basis=basis, charge=charge, spin=spin)
    # Get the problem from the driver
    problem = driver.run()

    # Apply active space reduction
    if active_orbitals is not None:
        problem = active_space(problem, active_orbitals)
    
    hamiltonian = problem.hamiltonian

    # Map the Hamiltonian to a qubit operator
    mapper = JordanWignerMapper()
    observables = mapper.map(hamiltonian.second_q_op())

    # Vuilding the Ansatz
    ansatz = build_ansatz(problem, mapper)

    # getting the constants from frozen orbitals and core repulsion
    if active_orbitals is not None:
        core_repulsion = hamiltonian.nuclear_repulsion_energy
        frozen_orbitals = hamiltonian.constants.get("ActiveSpaceTransformer", 0.0)
        constants = core_repulsion + frozen_orbitals
    else:
        constants = hamiltonian.nuclear_repulsion_energy
    
    if p_dep is not None:
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(
            depolarizing_error(p_dep, 2), ["cx"]
        )
        
        estimator = EstimatorV2(options={
        "backend_options": {"method": "density_matrix", "noise_model": noise},
        }
        )
    else:
        estimator = EstimatorV2()

    results = []
    #running the VQE optimization
    max_attempts = 3*trials  # safety limit to prevent infinite loops
    attempt = 0

    while len(results) < trials and attempt < max_attempts:
        attempt += 1
        init_params = 1e-8 * np.random.random(ansatz.num_parameters)

        try:
            res = minimize(
                cost_func,
                init_params,
                args=(ansatz, estimator, observables, shots),
                method=optimizer,
                options={'maxiter': 1000}
            )
            if not res.success:
                continue

            results.append(res.fun + constants)

        except Exception as e:
            print(f"[Attempt {attempt}] Exception during optimization: {e}")
            continue
    
    mean = np.mean(results)
    std = np.std(results)
    print(f"VQE results: mean = {mean}, std = {std}, trials = {trials}")
    return mean, std

vqe_pes = PesFunction(
    kernel_vqe, 
    {
        'basis': 'ccpvdz', # 
        'charge': 0, 
        'spin': 0, 
        'active_orbitals': [1, 2, 3, 4, 5], # indexing with 0, so 4 orbitals are filled one virtual is left
        'trials': 1, # if we are expecting poor convergence, we can increase the trials to find a mean value
        'optimizer': 'COLYLA',
        'shots': None, # exact results
        'p_dep': None  # probability of depolarization error
    }
)

# wrapping the pesfunction to take in variables
def vqe_pes_function(basis = 'ccpvdz', 
                    charge = 0, 
                    spin = 0, 
                    active_orbitals = [1,2,3,4,5], 
                    trials = 1, 
                    optimizer = 'COBYLA', 
                    shots = None, 
                    p_dep = None, 
                    **kwargs
                    ):
    return PesFunction(
        kernel_vqe,
        {
            'basis': basis,
            'charge': charge,
            'spin': spin,
            'active_orbitals': active_orbitals,
            'trials': trials,
            'optimizer': optimizer,
            'shots': shots,
            'p_dep': p_dep,  # probability of depolarization error
            **kwargs,  # allow additional parameters to be passed
        }
    )
