from setuptools import setup, find_packages

# TODO: replace the first four fields and 'ptt' by the name of the root folder

setup(
    name='pytorch_template',
    version='0.1',
    description='A Pytorch project template',
    url='https://github.com/camgbus/pytorch_template',
    keywords='python setuptools',
    packages=find_packages(include=['ptt', 'ptt.*']),
)