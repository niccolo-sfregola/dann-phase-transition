# Domain Adaptation Methods for Quantum Phase Transitions


This project investigates quantum phase transitions in 2D Ising models with random local magnetic fields using deep learning techniques. Monte Carlo simulations were used to generate spin configurations across different temperatures and disorder strengths.

Two approaches were compared: a standard Convolutional Neural Network (CNN) and a Domain-Adversarial Neural Network (DANN). The adversarial model leveraged knowledge from the zero-field case to generalize to disordered target domains and estimate critical temperatures under non-zero magnetic fields. 

Both models produced consistent phase diagrams, while the DANN showed stronger domain generalization and more accurate transition estimates in challenging regimes.
