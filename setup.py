from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="thompson-sampling",
    version="0.0.4",
    description="Thompson Sampling",
    author="Anton1o-I",
    author_email="a.iniguez21@gmail.com",
    packages=find_packages(),
    license="LICENSE.txt",
    url="https://github.com/Anton1o-I/thompson-sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["typing", "numpy", "seaborn", "matplotlib", "pandas"],
    tests_require=["pytest"],
)
