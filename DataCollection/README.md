Cancer_tp53_miner

This script is designed to extract TP53 biomarker information and current clinical trials and drugs.  

Usage:

cancer_tp53_miner.py

Output

Dumps data using implicit pathways  ../../sandbox/DC_env.  A v4 uuid folder is created with below structure

Dir:uuid ------> file:data.json 

Dir:uuid ------> Dir:Images ------> file:*png



This tool uses Selenium as a main component for mining the data.  

cancer_tp53_miner.py now deprecated - no need to use


#####new suite of software with  unit tests

->download;

    webscraper.py
    mycancergenome_scraper.py
    test_mycancergenome.py 

Install libraries:

Selenium
uuid
time
urllib.request 


Usage:

python mycancergenome_scraper.py 

This will write a data.json file based on the below relative path  from where you ran this script.  It contains a mined information focussed on TP53 for now

..\..\sandbox\DC_env\uuid(file)

This can also be download from pypi at; 
https://pypi.org/project/TP53-clinical-trials/

Unit Test:

Uses python unittest class to assert  MyCancerGenome methods.

###


Software fully dockerised and available from Docker Hub on request.
This tool run has been fully tested on an EC2 instance utilising AWS RDS and S3 for storing data and files, respectively.  

Usage: 
docker run  repositoryname/my-cancer-genome:mycancergenome


