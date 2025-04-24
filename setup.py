from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="tap30-ride-demand",
    version="0.1.0",
    author="Hamed",
    packages=find_packages(),
    install_requires=requirements,
)
