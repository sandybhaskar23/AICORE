from setuptools import setup
from setuptools import find_packages

setup(
    name = 'TP53 clinical trials',
    version = '0.0.1',
    description='This package extracts TP53 data consisting of genomic changes, disease and existing clinical drug trials and the phase of these trials',
    url='https://github.com/sandybhaskar23/AICORE/tree/main/DataCollection',
    author='Sanjeev Bhaskar',
    license= 'MIT',
    packages=find_packages(),
    install_requires= ['Selenium', 'uuid' ,'time', 'urllib.request','MyCancerGenome','boto3','botocore','webscraper','sqlalchemy'],
)