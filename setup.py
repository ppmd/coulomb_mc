from setuptools import setup, find_packages

long_description = """coulomb_mc is a FMM-MC implementation that is built on PPMD."""

install_requires = []
with open('requirements.txt') as fh:
    for l in fh:
        if len(l) > 0:
            install_requires.append(l)



setup(
   name='coulomb_mc',
   version='1.0',
   description='FMM-MC implementation',
   license="GPL3",
   long_description=long_description,
   author='William R Saunders',
   author_email='W.R.Saunders@bath.ac.uk',
   url="https://github.com/ppmd/coulomb_mc",
   packages=find_packages(),
   install_requires=install_requires,
   scripts=[],
   include_package_data=True
)
