# Jacobi Heat Diffusion

Heat diffusion is a famous equation used to calculate how temperature spreads from a warm point to all surrounding areas. While mathematicians have solved this problem analytically using differential equations, calculating exact solutions in real-world scenario, especially on grids with millions of point is computationally impossible.

To solve this, computer scientists rely on the **Jacobi algorithm**, an iterative method that approximates the heat equation's outcome across discrete grids.

---

## 📁 Repository Structure

### [`/01`](./01) — Parallel Computing Implementations
This folder contains various approaches to parallelizing the Jacobi algorithm across CPU and GPU architectures:
* Multi-threading strategies for multi-core CPUs.
* GPU acceleration techniques to distribute computation across massive parallel grids.

### [`/02`](./02) — GPU Hyperparameter Optimization
This folder contains experimental setups and benchmarks for optimizing performance on GPU architectures:
* Grid and block size hyperparameter searches.
* Performance tuning to achieve reliable, high-precision results while minimizing the execution time.

---

## 🚀 Key Concepts
* **Heat Equation Approximation:** Discretizing spatial heat distribution over large-scale matrix grids.
* **Jacobi Iterative Method:** Updating grid cells iteratively using neighboring values until convergence.
* **Parallel Optimization:** Benchmarking hardware performance across CPU threads and GPU thread blocks.
