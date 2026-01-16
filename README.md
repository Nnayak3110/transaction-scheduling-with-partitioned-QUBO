# transaction-scheduling-with-partitioned-QUBO

## Paper Overview

This repository contains the reference implementation and experimental artefacts for the paper:

Nitin Nayak, Alexandru Prisacaru, Umut Çalıkyılmaz, Jinghua Groppe, and Sven Groppe. 2025. Quantum-Enhanced Transaction Scheduling with Reduced Complexity via Solving QUBO Iteratively using a Locking Mechanism. In Proceedings of the 2nd Workshop on Quantum Computing and Quantum-Inspired Technology for Data-Intensive Systems and Applications (Q-Data '25). Association for Computing Machinery, New York, NY, USA, 26–35. https://doi.org/10.1145/3736393.3736701 ￼

The paper studies the transaction scheduling problem under conflict constraints and introduces a hybrid quantum–classical optimisation approach based on iteratively solving reduced QUBO subproblems using a locking mechanism. The goal is to significantly reduce problem complexity while preserving optimality, making the problem more accessible to near-term quantum devices and simulators.

## Key Ideas

Transaction scheduling is formulated as a Quadratic Unconstrained Binary Optimisation (QUBO) problem. While direct QUBO formulations grow rapidly in size and quickly exceed current quantum hardware limits, this work introduces a locking-based iterative strategy:
	•	At each iteration, a subset of transactions is locked at valid start times.
	•	The remaining problem is reformulated as a reduced iter-QUBO.
	•	Solving multiple smaller iter-QUBOs replaces solving one large QUBO.
	•	This leads to a 60–85% reduction in active binary variables in many cases, without sacrificing solution quality.

This approach provides a practical bridge between realistic database scheduling workloads and today’s NISQ-era quantum devices

## Contribution

The main contributions of this work are:
	•	A generalised QUBO formulation for transaction scheduling with conflict constraints
	•	A locking mechanism that systematically partitions the search space into valid subproblems
	•	Formal criteria to determine the number of required iterations
	•	A hybrid evaluation using:
	•	Simulated Annealing (D-Wave Ocean SDK)
	•	Gate-based quantum simulation with QAOA
	•	An extensive experimental study showing significant reductions in QUBO size and runtime compared to full formulations


## Citation

```bibtex
@inproceedings{10.1145/3736393.3736701,
author = {Nayak, Nitin and Prisacaru, Alexandru and \c{C}al\i{}ky\i{}lmaz, Umut and Groppe, Jinghua and Groppe, Sven},
title = {Quantum-Enhanced Transaction Scheduling with Reduced Complexity via Solving QUBO Iteratively using a Locking Mechanism},
year = {2025},
isbn = {9798400719448},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3736393.3736701},
doi = {10.1145/3736393.3736701},
abstract = {In this work, we investigate the problem of transaction scheduling under conflict constraints and present a hybrid quantum-classical approach to its optimisation. Our proposed algorithm uses the concept of locking the transaction to solve QUBO iteratively, which explores the search space over the different sets of transactions for efficient scheduling. Experiments have been conducted using gate-based quantum simulators with QAOA and classical solvers such as simulated annealing. Our findings suggest that reduced problem formulations (via locking mechanism) for a certain level of problem complexity significantly enhance the solvability of scheduling instances, potentially making them accessible to near-term quantum devices.},
booktitle = {Proceedings of the 2nd Workshop on Quantum Computing and Quantum-Inspired Technology for Data-Intensive Systems and Applications},
pages = {26–35},
numpages = {10},
keywords = {QUBO, subQUBO, Transaction Scheduling, Quantum Optimization},
location = {
},
series = {Q-Data '25}
}
