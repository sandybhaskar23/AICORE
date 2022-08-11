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




