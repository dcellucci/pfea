try:
    import setuptools
except ImportError:
    pass

from distutils.core import setup

setup(name='pfea',
	version='1.0',
	description='Tools for Generating and Simulating Cellular Solids',
	author='Daniel Cellucci',
	url='https://github.com/dcellucci/pfea',
	package_dir={'pfea': 'src'},
	packages=['pfea','pfea.geom','pfea.util'],
	install_requires=[
		'numpy',
		'scipy',
		'networkx',
		'matplotlib',
		'cvxopt',
	]
	)