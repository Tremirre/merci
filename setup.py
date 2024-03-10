from setuptools import setup, find_packages

# get version from git tag
from subprocess import check_output

version = check_output(["git", "describe", "--tags"]).strip().decode().lstrip("v")

setup(packages=find_packages(), install_requires=[], version=version)
