from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "networkx>=2.1",
    "numpy>=1.8.1",
    "scikit-learn>=0.19.1",
    "scipy>=1.1.0",
    "seaborn>=0.9.0",
]

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Modeling the Drosophila larva connectome",
    author="Neurodata",
    license="BSD-3",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=[
        "git+https://github.com/neurodata/graspy.git#egg=dros",
        "git+https://github.com/neurodata/mgcpy.git#egg=master",
    ],
)
