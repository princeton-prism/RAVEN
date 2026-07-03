from setuptools import setup, find_packages

setup(
    name="qqmm",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['configs/**/*']},
)
