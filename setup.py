from setuptools import setup, find_packages

setup(
    name="thompson-sampling",
    version="0.0.1",
    description="Thompson Sampling",
    author="Anton1o-I",
    author_email="a.iniguez21@gmail.com",
    packages=find_packages(),
    license="LICENSE.txt",
    long_description=open("README.md").read(),
    install_requires=["typing", "numpy", "seaborn", "matplotlib", "pandas"],
    tests_require=["pytest"],
)

