from __future__ import unicode_literals
from distutils.log import error
from telnetlib import DET
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
import time
import json
import urllib.request 
import boto3
import webscraper
import pandas as  pd
import uuid
import mcg_db
""" Creates a scraper for  TP53 the guardian of the genome.  The selenium driver reads the pages and scrolls till it captures all the card details
    Data and images are extracted to local storage as json and png
"""

class GuardianScarper:


    def __init__(self,url ="https://www.mycancergenome.org/content/biomarkers/#search=TP53"):
           
        self.driver = webdriver.Chrome() 
        self.driver.get(url)
        ##call class 
        self.WS = webscraper.WebScraper(self.driver)
        ##list     
        self.link_list =[]
        self.clintrials_link =[]
        ##get v4 uuid
        self.WS.get_uuid()
        ##build the data
        self.summary_det={}
        ##no longer needed since using uuid5 for actual links 
        #self.summary_det['uuid'] = self.WS.uuid
        ##aws handle
        self.s3 = boto3.client('s3')

        ##global defined key for dict needed for clinical trials details
        self.cct= 'ClinicalTrialsTitle'
        
        

    def get_links(self, val='small-up-1 medium-up-2 large-up-2', key=None)-> list:
        """ Finds the href links in web pages
            Args:
                driver: webdriver.Edge
                The driver that contains information about the current page

                driver              : Uses contains to deal with slight variance in html class name
                link_list (dict)    : Stores all the the href links 
                summary_det (dict)  : Stores the extracted data 
            
            Returns:   
                link_list
                    A list with all the links in the page
                summary_det
                    Stored extracted data
        """
        ##has to be class directly parent of rest of listings
        ##need to be agnostic to nuance naming changes   
        #  WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, '//*[@id="gdpr-consent-notice"]')))    
        #WebDriverWait(self.driver.implicitly_wait(),18).until(EC.visibility_of_element_located((By.XPATH,"//div[contains(@class,'{val}')]"))) < does not work  
        #self.driver.implicitly_wait(18) < -does not resolve issue
        #self.driver.page_source    < does resolve issue                  
        tp53_container = self.driver.find_element(by=By.XPATH, value=f"//div[contains(@class,'{val}')]")
        #input[contains(@id,'id')]
        tp53_list = tp53_container.find_elements(by=By.XPATH, value='//div[@class="card-section"]')
       
        print(f'Current Url is: {self.driver.current_url}')
        if key is not None:            
            self.summary_det[key][self.cct] = {}

        for tp53_property in tp53_list:
            a_tag = tp53_property.find_element(by=By.TAG_NAME, value='a')
            link = a_tag.get_attribute('href')
            #print('in links')
            #print(link)
            uniqueid = str(uuid.uuid5(uuid.NAMESPACE_DNS,link))
            #print (uniqueid)
            h5_tag = tp53_property.find_element(by=By.TAG_NAME, value='h5')
            det = h5_tag.get_attribute('title')
            
            ###get card details while here            
            chunks = tp53_property.text.split('\n')                 

            ###convert rest in list into dictionary
            _res_dct = {chunks[i].split(':')[0].replace(' ','_'): chunks[i].split(':')[-1]  for i in range(1, len(chunks)) if ':' in chunks[i]}
            ##1st condition is based on sub links now being called. This will now store the sub information under the initial group key
            if key is not None:
                tri = type(self.summary_det[key][self.cct])
                #print(tri)
                ##create a nested dictionary. This happens after main key has created therefore update is enough
                self.summary_det[key][self.cct].update({ uniqueid : {'Link':link, 'ClinicalTrialDescription': det, 'Id': key}})
                #self.summary_det[key][self.cct][det].update({'link':link})
                self.summary_det[key][self.cct][uniqueid].update(_res_dct)                
            else:
                ##store infor from 1st group of links and their relevant information                     
                self.summary_det[uniqueid] = {'Link':link, 'Biomarkers' : det }
                self.summary_det[uniqueid].update(_res_dct)
            #print(uniqueid)
            self.link_list.append(f'{uniqueid}:{link}')                           
            
        ##high degree of redundancy       
        self.link_list = list(set(self.link_list))

        print(f'There are {len(self.link_list)} properties in this page')
        ###needs to be returned to the function calling this class method
        return self.link_list
   

    def get_link_details(self, big_list):    
        """ Extracts all the details of links i.e: disease, drugs from the aggregated stored links dictionary
        
            Args: 
                driver                              : webdriver.Chrome get function for link
                get_image (method)                  : Trigger the extraction of href links for images
                clinicaltrials_link (list)          : unique set of links to the clinical trials page
                get_clinical_trial_details (Method) : Triggers the extractions  of clinical details

            Returns:
                    Call to clinicaltrials_link results in 
                    summary_det (dict) being updated

        
        """
        for link in big_list:
            ##need to make sure the key is split and passed on
            k,lnk = link.split(':',1)

            ##now in the card details i.e TP53 nonsense ...            
            self.driver.get(lnk)
            ##in this case there is no images bar the logo and pathway need to traverse links to get them
            ###there is only one pathway map therefore do not store muliple times
            self.WS.get_image(self.driver)
            
            try:
                clinical_trials = self.driver.find_element(by=By.PARTIAL_LINK_TEXT, value = 'View Clinical Trials')               
                self.clintrials_link.append(f"{k}:{clinical_trials.get_attribute('href')}")    
                #print(clintrials_link)
            except:
                ###if not present then not an issue not all variance has clinical trials
                print('moving on')
                continue
        ##make unique
        self.clintrials_link = list(set(self.clintrials_link))            

            ###now get the clinical trials details. 
        self.get_clinical_trial_details(self.clintrials_link)

    

    def get_clinical_trial_details(self,clintrials_link = ["https://www.mycancergenome.org/content/clinical_trials/#alterations=TP53%20Mutation"]):

        """ Extracts all the details of clinical trials and call the get_links

            Args:
                driver                              : webdriver.Chrome get function for link
                scroll_down (Method)                : Scrolls down page fully before dealing with cookies
                get_links (Method)                  : Call the method to extract the data from links
            
            Returns:
                Call to get_links results in 
                    summary_det (dict) being updated
        
        """

        print('clinical trials')
        #print (self.clintrials_link)
        for cln in clintrials_link:
            k,cln = cln.split(':',1)
            #print(k)
            
            print('In clinical trials details')
            #print (cln)
            #print (self.summary_det)
            ##need this to allow driver time to load the actual page. Misdirection technique
            self.driver.get('https://www.mycancergenome.org/content/biomarkers/#search=TP53')
            time.sleep(2)
            self.driver.get(cln)
            self.driver.implicitly_wait(1)       
            self.WS.scroll_down()
            time.sleep(1)
            #print (self.driver.page_source)
            ##reuse
            self.get_links(key=k)
            
   

    def download_images(self):

        """ Method to download images and rename based on order in list
            Not many images on the website. Pages visited only have logos

            Args:
                get_path (Method)   : Get the global data store relative path and created it, if it does not exist
                imgdir (str)        : Define a subfolder images within the global relative data store directory

        
            Returns:
                Results in the images being stored in the images folder which is relative to where the script is triggered. 
        
        """
        self.get_path()
        imgdir = Path(f"{self.p}\images")
        imgdir.mkdir(parents=True, exist_ok=True)    
        image_links = self.WS.get_image(self.driver)
        for i, img  in enumerate(image_links):
            ##store the images but rename using index.
            fimage = (img.split('/')[-1]).replace('.png','')       
            urllib.request.urlretrieve(img,(f"{imgdir}\{fimage}_{i}.png"))
            self.s3.upload_file (f'{imgdir}\{fimage}_{i}.png' ,'mycancergenome' , f'{fimage}_{i}.png')   
        
   
    def get_path(self):
        """ Define and make dirs for storing data

            Args:
                p (str) : path for data storage   
        
            Returns:
                p 
                    Called within class or external to class                
        
        """
        ##define the pathway for storing dtaa in it relative pathway
        ##makes it more compatible for any directory structure
        self.p = Path(f"..\..\sandbox\DC_env\{str(self.WS.uuid)}")
        self.p.mkdir(parents=True, exist_ok=True)

        return self.p 




def deepmine(url="https://www.mycancergenome.org/content/biomarkers/#search=TP53"):
    """ Starts of the data extraction of mycancergenome for TP53 data

        Args:
            guardscap (class object)            : Handle for all methods within the GuardianScraper class
            big_list (dict)                     : Aggregated link results from mining the webpages
            get_link_details (guardscap Method) : extract data from big_list links
            get_path (guardscap Method)         : sets the relative path for storing data 
            p (str)                             : pathway relative path
            download_images (guardscap Method)  : Extract the images 
    
        Returns:

            Nothing the job is done
    
    """


    ##get the json file ready

    guardscap= GuardianScarper(url)  
    guardscap.WS.load_and_accept_cookies()

    big_list = []
    for i in range(1): # The first 5 pages only
        #I am only human
        
        big_list.extend(guardscap.get_links()) # Call the function we just created and extend the big list with the returned list
  
        print(f'There are {len(big_list)} tp53 groups so far')   
        time.sleep(3)

    ##pack the list back to guardian let it handle it
    guardscap.get_link_details(big_list)


    ##dump this data to v4 uuid folder
    p = guardscap.get_path()  
    
    ##write the dictioary data into a json file
    with (p / 'data.json').open('w') as opened_file:          
          json.dump(guardscap.summary_det,opened_file)

    df = pd.DataFrame(guardscap.summary_det)
    df.transpose().to_csv(r'mycancergenome_data.csv',  mode='a')
    mcg_db.McsInterface(guardscap.summary_det, guardscap.cct).run()
    #print(df)

    guardscap.s3.upload_file(f'{p}/data.json' ,'mycancergenome', 'data.json')
    ##download the images
    guardscap.download_images()

    ##neatly close the selnium chrome driver
    guardscap.driver.quit()
        


if __name__ == '__main__':
    ##set the global link for now   
    deepmine()