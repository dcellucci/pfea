try:
    import setuptools
except ImportError:
    pass

from setuptools import setup

setup(name='pfea',
	version='1.0',
	description='Tools for Generating and Simulating Cellular Solids',
	author='Daniel Cellucci & Nick Cramer',
	url='https://github.com/dcellucci/pfea',
	packages=['pfea','pfea.geom','pfea.util'],
	install_requires=[
		'numpy',
		'scipy',
		'networkx',
		'matplotlib',
		'cvxopt',
	]
	)
