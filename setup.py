from setuptools import setup, find_packages

setup(
    name='geometric_data',
    version='0.2',
    description='A project that deals mainly with geometric dummy data',
    url='https://github.com/camgbus/geometric_data',
    keywords='python setuptools',
    packages=find_packages(include=['gd', 'gd.*']),
)