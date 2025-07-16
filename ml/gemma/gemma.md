
Bill Dally - only 20% can be parallelized

Control of CPU

Architect in GTC
- break down
- GTC talks python solution for CUDA
- 

Can't do general 

https://www.reddit.com/r/LocalLLaMA/comments/1kuy45r/gemma_3n_architectural_innovations_speculation/


Workload 80% can't be parallelized
100000 cuda cores

python 3.14 no GIL

262144 (vocabulary) x 256 x 35 (layers)

Paper suggestion:  XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models  https://arxiv.org/abs/2411.15100   
DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal https://arxiv.org/abs/2503.14269 
Pre3: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation https://arxiv.org/abs/2506.03887


https://openreview.net/pdf?id=GmqZ3WvkeV

Constrastive Learning
- feature extractedr
- distallation