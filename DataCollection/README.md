Cancer_tp53_miner - Now deprecated for mycancergenome_scraper.py. See below

This script is designed to extract TP53 biomarker information and current clinical trials and drugs.  

####Usage:

cancer_tp53_miner.py

#####Output

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

#####Install libraries:

Selenium
uuid
time
urllib.request 

also;
Install pgadmin4


###Usage:

python mycancergenome_scraper.py 

This will write a data.json file based on the below relative path  from where you ran this script.  It contains a mined information focussed on TP53 for now

..\..\sandbox\DC_env\uuid(file)

This can also be download from pypi at; 
https://pypi.org/project/TP53-clinical-trials/

Unit Test:

Uses python unittest class to assert  MyCancerGenome methods.


Software builds tables in postgresql and seeds scapered data.   It will add new data if available. 


Software fully dockerised and available from Docker Hub on request.
This tool run has been fully tested on an EC2 instance utilising AWS RDS and S3 for storing data and files, respectively.  

Prometheus and Grafana dashboard has been created to monitor docker  'state' and 'load' on the EC2 instance
Helpful pages for grafana setup is:
https://www.radishlogic.com/aws/ec2/how-to-install-grafana-on-ec2-amazon-linux-2/ 

Use Grafana home page for more details
https://grafana.com/grafana/ 


Continuous integration set up using github actions.  Software can be downloaded;
docker image  pull  sandybhaskar23/my-cancer-genome:mycancergenome

All authentication variables have now been moved to using .env files which must reside in your user home space consisting of 
 
 .env file 
HOST=aws_ec2  
PASSWORD=password  
PORT=5432  
USRNAME=postgres  
DATABASE=db_name  
aws_access_key_id=KEY  
aws_secret_access_key=SECRET_KEY  

##Usage:   
docker run  repositoryname/my-cancer-genome:mycancergenome










