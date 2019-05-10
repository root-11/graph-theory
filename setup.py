"""
graph-theory
"""
from datetime import datetime
from setuptools import setup
from pathlib import Path

v = datetime.now()
__version__ = "{}.{}.{}.{}".format(v.year, v.month, v.day, v.hour * 3600 + v.minute*60 + v.second)

folder = Path(__file__).parent
file = "README.md"
readme = folder / file
assert isinstance(readme, Path)
assert readme.exists(), readme
with open(str(readme), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="graph-theory",
    version=__version__,
    url="https://github.com/root-11/graph-theory",
    license="MIT",
    author="Bjorn Madsen",
    author_email="bjorn.madsen@operationsresearchgroup.com",
    description="A graph library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="graph graphs optimization tsp shortest-path",
    packages=["graph"],
    include_package_data=True,
    data_files=[(".", ["LICENSE", "README.md"])],
    platforms="any",
    install_requires=[],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)


