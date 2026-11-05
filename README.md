# cpu-vector-8X-search-engine

What this is
This repo is a single-notebook proof of concept for an ultra-fast CPU vector search engine designed for RAG-style retrieval and recommendation workloads. Rather than relying on a full app framework, it implements a custom NumPy-based vector database and similarity pipeline, then benchmarks it against a standard baseline to show the speedup.

Stack
Language(s): Python
Framework / runtime: Jupyter Notebook (Python 3.13)
Notable libraries: NumPy, SciPy-like numeric patterns, no app framework or web server
How it's organized
Text
LICENSE                  MIT license
README.md                minimal repo stub
cpu-vector-search-engine.ipynb
                        full implementation, benchmarks, and demo
How it fits together: Everything lives in one notebook. The notebook defines custom kernels for inverse-square-root normalization, cosine similarity search, a simulated embedding database, and a retrieval demo that shows top-k document matches. It also includes benchmark sections comparing a baseline approach against the optimized engine.

How to run it
Because this repo has no package manifest or app entrypoint, the practical path is to run the notebook in a Python/Jupyter environment.

bash
python -m venv .venv
source .venv/bin/activate
pip install numpy jupyter
jupyter lab cpu-vector-search-engine.ipynb
Then execute the cells in order. The notebook itself contains:

the custom vector-normalization engine
a simulated vector database
baseline vs optimized speed comparisons
an end-to-end retrieval example
There are no obvious secrets or environment variables to configure.

What the notebook is doing
The implementation is built around a custom “ultra-optimized” CPU search pipeline:

It normalizes embeddings in-place to avoid intermediate allocations.
It uses a custom inverse-square-root kernel inspired by fast numeric tricks.
It computes cosine similarity and returns top-k matches efficiently.
It benchmarks the optimized path against a slower NumPy baseline.
It demonstrates retrieval on synthetic embeddings and a “ground truth” document index.
In other words, this is a performance-exploration repo more than a production service: it is meant to show how to squeeze more throughput out of CPU-only vector search using careful numerical and memory techniques
