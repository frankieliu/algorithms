adagrad
- adaptive per parameter
2014 adam
- momentum
adamw
- separate weight decay
muon

zero, one, two
- different amount of distributed training
- Zero Redundancy Optimizer (ZeRO) is a memory optimization method developed by Microsoft’s DeepSpeed to enable efficient distributed training (and inference) of very large models. Instead of replicating all model states (parameters, optimizer states, and gradients) on each GPU, ZeRO intelligently partitions them across devices—dramatically reducing memory redundancy without sacrificing throughput or requiring model code changes

M := mu M + \grad L(W)
O := Newton-Schulz(M)
W := W - eta O

Approximately solve MM^T -1/2 M = UV^T

# From gemini summary:
The Newton-Schulz orthogonalization is a family of iterative algorithms used to transform a given matrix such that its rows or columns form an orthonormal set of vectors. In simpler terms, it's a method to "orthogonalize" a matrix.

Here's a breakdown of key aspects:

* **Symmetric Orthogonalization:** Unlike methods like Gram-Schmidt, which sequentially orthogonalize vectors, Newton-Schulz treats all rows/columns symmetrically, meaning no particular vector is singled out in the process.

* **Approximating the Orthogonal Factor:** The goal of Newton-Schulz orthogonalization is often to approximate the orthogonal factor ($U V^T$) of a matrix's singular value decomposition (SVD) ($M = U \Sigma V^T$). This is essentially "snapping the singular values of $\Sigma$ to one," while preserving zero singular values as zero.

* **Odd Matrix Polynomials:** The iterations are typically based on odd matrix polynomials of the form:
    $p(X) = aX + bXX^TX + c(XX^T)^2X + \dots$
    A crucial property of these polynomials is that they commute with the SVD, meaning $p(U \Sigma V^T) = U p(\Sigma) V^T$. This allows the iteration to act on the singular values directly.

* **Convergence to the Sign Function:** For certain coefficients and conditions, iterating these polynomials can cause the scalar polynomial $f(x)$ to approximate the sign function ($sgn(x)$) on the singular values. When applied to the matrix, this effectively orthogonalizes it by transforming the singular values towards 1 (for non-zero values) and -1, or maintaining 0. For example, a common cubic iteration is $f(x) = \frac{3}{2}x - \frac{1}{2}x^3$.

* **Applications:** Newton-Schulz iterations are useful in various fields, particularly in machine learning and numerical linear algebra:
    * **Neural Network Optimization:** They have been used for enforcing orthonormality of weight matrices in neural networks, such as in the Muon optimizer. This can help with stability and performance during training.
    * **Efficient Retraction for Riemannian Optimization:** They provide an efficient alternative to traditional methods for projecting onto the Stiefel manifold (the set of orthogonal matrices) in Riemannian optimization.
    * **Matrix Sign Function Computation:** The Newton-Schulz iteration is also a quadratically convergent, inversion-free method for computing the matrix sign function, which has applications in electronic structure calculations and control theory.

* **Computational Efficiency:** A significant advantage of Newton-Schulz iterations is that they primarily rely on matrix multiplications, making them computationally efficient, especially on hardware like GPUs.

* **Convergence Conditions:** Convergence of the standard Newton-Schulz method for orthogonalization typically requires the singular values of the initial matrix to lie within a certain range (e.g., $(0, \sqrt{3})$). Variations and scaling techniques exist to improve convergence and handle matrices with wider ranges of singular values.

In summary, the Newton-Schulz orthogonalization is a powerful iterative technique for transforming a matrix into an orthogonal or approximately orthogonal form, leveraging properties of matrix polynomials and singular value decomposition.

# Amgad Hasan
moonshot-k2-blog

# Architecture
github.com/AmgadHasan/moonshot-k2-blog/blob/main/README.md
Sebastian Rashka

# Agentic Intelligence
1. model should be intelligent to know which tool to use
1. RL:
   1. algorithm
   1. environment
   1. priors
1. post training
   1. alpha proof - self generated
   1. verifiable problems in RL
1. muonclip optimizer
   1. muon (keller jordan)
   1. reduce # heads
   1. training exploding attention logits
   1. adaptive factor soft capping
1. large scale agentic data synthesis
   1. evolve / instruct algorithm
   1. rubric based tasks

# general reinforcement learning

1. code and solution
1. math solution
1. how to expand into other tasks

Sebastian Raschka
translation of Shaowei Liu's writing

Eugene Cheah
(Building the transformer killer
attention-free ai model
wiki.rwkv.com)


