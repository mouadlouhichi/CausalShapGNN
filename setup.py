from setuptools import setup, find_packages

setup(
    name="causalshapgnn",
    version="1.0.0",
    author="Mouad LOUHICHI",
    author_email="mouad_louhichi@um5.ac.mam",
    description="Causal Disentangled GNN with Topology-Aware Shapley Explanations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mouadlouhichi/CausalShapGNN",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "requests>=2.26.0",
    ],
)
