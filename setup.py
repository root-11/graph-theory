"""
graph-theory
"""
from setuptools import setup
from pathlib import Path


root = Path(__file__).parent

__version__ = None
version_file = root / "graph" / "version.py"
exec(version_file.read_text())
assert isinstance(__version__, str)  # noqa

with open(root / "README.md", encoding="utf-8") as f:
    long_description = f.read()

with open(root / "requirements.txt", "r", encoding="utf-8") as fi:
    requirements = [v.rstrip("\n") for v in fi.readlines()]

keywords = list(
    {
        "complex-networks",
        "discrete mathematics",
        "graph",
        "Graph Theory",
        "graph-algorithms",
        "graph-analysis",
        "analysis",
        "algorithms",
        "graph-generation",
        "graph-theory",
        "graph-visualization",
        "graphs",
        "math",
        "Mathematics",
        "maths",
        "generation",
        "generate",
        "theory",
        "minimum-spanning-trees",
        "network",
        "Networks",
        "optimization",
        "python",
        "shortest-path",
        "tsp",
        "tsp-solver",
        "minimum",
        "spanning",
        "tree",
        "assignment problem",
        "flow-problem",
        "hash",
        "graph-hash",
        "random graph",
        "search",
        "cycle",
        "path",
        "flow",
        "path",
        "shortest",
        "component",
        "components",
        "adjacency",
        "matrix",
        "all pairs shortest path",
        "finite state machine",
        "fsm",
        "adjacent",
        "pairs",
        "finite",
        "state",
        "machine",
        "traffic-jam",
        "traffic-jam-solver",
        "solver",
        "hill-climbing",
        "simple",
        "simple-path",
        "critical",
        "path",
        "congestions",
        "jam",
        "traffic",
        "optimisation",
        "method",
        "critical-path",
        "minimize",
        "minimise",
        "optimize",
        "optimise",
        "merkle",
        "tree",
        "merkle-tree",
        "hash-tree",
    }
)

keywords.sort(key=lambda x: x.lower())


setup(
    name="graph-theory",
    version=__version__,
    url="https://github.com/root-11/graph-theory",
    license="MIT",
    author="https://github.com/root-11",
    description="A graph library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=keywords,
    packages=["graph"],
    python_requires=">=3.7",
    include_package_data=True,
    data_files=[(".", ["LICENSE", "README.md", "requirements.txt"])],
    platforms="any",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
