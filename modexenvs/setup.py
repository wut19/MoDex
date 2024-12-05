from setuptools import find_packages
from distutils.core import setup

setup(
    name='modexenvs',
    version='1.0.0',
    author='Tong Wu',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='Environments for ',
    install_requires=['isaacgym',
                      'myosuite',
                      'stable-baselines3']
)