from selenium import webdriver
from selenium.webdriver.common.by import By
from pathlib import Path
import time
import json
import urllib.request 
import webscraper

""" Creates a scraper for  TP53 the guardian of the genome.  The selenium driver reads the pages and scrolls till it captures all the card details
    Data and images are extracted to local storage as json and png
"""

class GuardianScarper:


    def __init__(self,url ="https://www.mycancergenome.org/content/biomarkers/#search=TP53"):
           
        self.driver = webdriver.Edge() 
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
        self.summary_det['uuid'] = self.WS.uuid
        
        

    def get_links(self, val='small-up-1 medium-up-2 large-up-2')-> list:
        """ Finds the href links in web pages
            Args:
                driver: webdriver.Chrome
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
        tp53_container = self.driver.find_element(by=By.XPATH, value=f"//div[contains(@class,'{val}')]")
        #input[contains(@id,'id')]
        tp53_list = tp53_container.find_elements(by=By.XPATH, value='//div[@class="card-section"]')

        for tp53_property in tp53_list:
            a_tag = tp53_property.find_element(by=By.TAG_NAME, value='a')
            link = a_tag.get_attribute('href')
       
           
            self.link_list.append(link)
            ###get card details while here
            chunks = tp53_property.text.split('\n')     
            ##store in case I forget                       
            self.summary_det[chunks[0]] = {'link':link}
            ###convert rest in list into dictionary
            _res_dct = {chunks[i].split(':')[0]: chunks[i].split(':')[-1] for i in range(1, len(chunks))}
            
            self.summary_det[chunks[0]].update(_res_dct)
            #print(self.summary_det)                

        print(f'There are {len(self.link_list)} properties in this page')
        ###needs to be returned to the function calling this class method
        return self.link_list



    def get_summary(self, tp53_property,card_title)-> dict:
        """ Get summary details from each card.  
            Each card holds variant coorindate details

            Args:
                driver              : webdriver.Chrome
                tp53p_list (list)   : Elements of p tag from html
                innertext (str)     : The values from elements of tp53_list and tp53p_list
                span_tag (str)      : 1st index values of split from innertext
                inner_text (str)    : 2nd indexed value of split from innertext
                summary_det (dict)  : Stores the extracted data 

            Returns:
                summary_det
                    Stored extracted data
        """


         ##has to be class directly parent of rest of listings
        tp53_container = tp53_property.find_element(by=By.XPATH, value='//*[@class="news-card-article"]')
        tp53_div = tp53_container.find_element(by=By.XPATH, value='./div')
        tp53_list = tp53_div.find_elements(by=By.XPATH, value='./p')
        ##tidy up naming = based on structure of tp53_container need a diff handle
        tp53p_list = tp53_container.find_elements(by=By.XPATH, value='./p')
        
        for tp53_prop in tp53_list:            
            ###get the key for this 1st div
            _innertext = tp53_prop.text
            ###add to dictionary
            span_tag, inner_text = _innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }           
        

        ###need to get the p tags now            
        for tp53p_prop in tp53p_list:
           
            ###get the inner text encapsulated as a react text. 
            _innertext= tp53p_prop.text
            ###add to dictionary
            span_tag, inner_text = _innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }
                 
    
        #time.sleep(30)   

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
        for lnk in big_list:
            ##now in the card details i.e TP53 nonsense ...
            self.driver.get(lnk)
            ##in this case there is no images bar the logo and pathway need to traverse links to get them
            ###there is only one pathway map therefore do not store muliple times
            self.WS.get_image(self.driver)
            
            try:
                clinical_trials = self.driver.find_element(by=By.PARTIAL_LINK_TEXT, value = 'View Clinical Trials')               
                self.clintrials_link.append(clinical_trials.get_attribute('href'))    
                #print(clintrials_link)
            except:
                ###if not present then not an issue not all variance has clinical trials
                continue
        ##make unique
        self.clintrials_link = list(set(self.clintrials_link))            

            ###now get the clinical trials details. 
        self.get_clinical_trial_details()

    

    def get_clinical_trial_details(self):

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
        for cln in self.clintrials_link:
            #print (cln)
            self.driver.get(cln)
            self.WS.scroll_down()
            time.sleep(4)
            #print (self.driver.page_source)
            ##reuse
            self.get_links()
            
   

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
        self.p = Path(f"..\..\sandbox\DC_env\{self.summary_det['uuid']}")
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

    ##download the images
    guardscap.download_images()

    ##neatly close the selnium edge driver
    guardscap.driver.quit()
        


if __name__ == '__main__':
    ##set the global link for now   
    deepmine()