from setuptools import setup, find_packages

# get version from git tag
from subprocess import check_output

version = check_output(["git", "describe", "--tags"]).strip().decode()

setup(packages=find_packages(), install_requires=[])
