from select import select
from turtle import title
from attr import attrib
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import time
import uuid
import json
from pathlib import Path
import urllib.request 

''' Creates a scraper for  TP53 the guardian of the genome.  The selenium driver reads the pages and scrolls till it captures all the card details
    Data and images are extracted to local storage as json and png
'''

class GuardianScarper:


    def __init__(self,URL):
           
        self.driver = webdriver.Edge() 
        self.driver.get(URL)
        ##list 
        self.image_links = []
        self.link_list =[]
        self.clintrials_link =[]
        ##get v4 uuid
        self.get_uuid()
        ##build the data
        self.summary_det={}
        self.summary_det['uuid'] = self.uuid
        



    def load_and_accept_cookies(self):
        
        """ Deals with any cookies and resolves scrolling down

            Args:
            driver: webdriver.Chrome
                The driver that contains information about the current page

                scroll_down.driver (Method) : Scrolls down page fully before dealing with cookies
                
                    (default is False)

            Returns:
                driver : cookies accepted 
        """

        #Open mycancer genome and accept the cookies      
        
        self.scroll_down()
        
        time.sleep(3) 
        ##get driver to wait 10 seconds
        delay = 10
        try:
            WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, '//*[@id="gdpr-consent-notice"]')))
            print("Frame Ready!")
            self.driver.switch_to.frame('gdpr-consent-notice')
            accept_cookies_button = WebDriverWait(self.driver, delay).until(EC.presence_of_element_located((By.XPATH, '//*[@id="save"]')))
            print("Accept Cookies Button Ready!")
            accept_cookies_button.click()
            time.sleep(1)
        except TimeoutException:
            print("Loading took too much time! to be expected no cookes at this website")
            ###there are no cookies consent request for this website
            pass
        except:
            pass

        

    def get_links(self):
        ''' Finds the href links in web pages
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
        '''
        ##has to be class directly parent of rest of listings
        ##need to be agnostic to nuance naming changes                               
        tp53_container = self.driver.find_element(by=By.XPATH, value="//div[contains(@class,'small-up-1 medium-up-2 large-up-2')]")
        #input[contains(@id,'id')]
        tp53_list = tp53_container.find_elements(by=By.XPATH, value='//div[@class="card-section"]')

        for tp53_property in tp53_list:
            a_tag = tp53_property.find_element(by=By.TAG_NAME, value='a')
            link = a_tag.get_attribute('href')
                #print ('start')
            #print (link)
            self.link_list.append(link)
            ###get card details while here
            chunks = tp53_property.text.split('\n')     
            ##store in case I forget                       
            self.summary_det[chunks[0]] = {'link':link}
            ###convert rest in list into dictionary
            res_dct = {chunks[i].split(':')[0]: chunks[i].split(':')[-1] for i in range(1, len(chunks))}
            
            self.summary_det[chunks[0]].update(res_dct)
            #print(self.summary_det)                

        print(f'There are {len(self.link_list)} properties in this page')
        ###needs to be returned to the function calling this class method
        return self.link_list



    def get_summary(self, tp53_property,card_title):
        ''' Get summary details from each card.  
            Each card holds variant coorindate details

            Args:
                driver              : webdriver.Chrome
                tp53_list (list)    : Elements of p tag from html
                tp53p_list (list)   : Elements of p tag from html
                innertext (str)     : The values from elements of tp53_list and tp53p_list
                span_tag (str)      : 1st index values of split from innertext
                inner_text (str)    : 2nd indexed value of split from innertext
                summary_det (dict)  : Stores the extracted data 

            Returns:
                summary_det
                    Stored extracted data
        '''


         ##has to be class directly parent of rest of listings
        tp53_container = tp53_property.find_element(by=By.XPATH, value='//*[@class="news-card-article"]')
        tp53_div = tp53_container.find_element(by=By.XPATH, value='./div')
        tp53_list = tp53_div.find_elements(by=By.XPATH, value='./p')
        ##tidy up naming = based on structure of tp53_container need a diff handle
        tp53p_list = tp53_container.find_elements(by=By.XPATH, value='./p')
        summary_det = {}
        for tp53_prop in tp53_list:            
            ###get the key for this 1st div
            innertext = tp53_prop.text
            ###add to dictionary
            span_tag, inner_text = innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }           
        

        ###need to get the p tags now            
        for tp53p_prop in tp53p_list:
           
            ###get the inner text encapsulated as a react text. 
            innertext= tp53p_prop.text
            ###add to dictionary
            span_tag, inner_text = innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }
                 
        #print(link_list)
        #time.sleep(30)

    def scroll_down(self):
        """Ensures HTML is fully loaded to  allow the whole page to be mined

            Args:
                driver: webdriver.Chrome scrolldown function
            
            Returns:
                Results in the webpage being fully loaded
                    
        """
        ##make sure the html is fully loaded
        self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")


    def get_link_details(self, big_list):    
        """Extracts all the details of links i.e: disease, drugs from the aggregated stored links dictionary
        
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
            self.get_image()
            
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
            self.scroll_down()
            time.sleep(4)
            #print (self.driver.page_source)
            ##reuse
            self.get_links()
            

    def get_image(self):

        """ Extracts all the links from the href tags

            Args:
                driver              : webdriver.Chrome get function for link
                image_links (dict)  : stores unique image links
        
            Returns:
                image_links 
                    Contains all the unique links found on the web pages loaded
        
        """
        ##logo
        ##html class name. 
        #htclass_name = ['nav-bar show-for-medium','associated-pathways']
        htimage = self.driver.find_elements(by=By.TAG_NAME, value = 'img')
        imlnk = []
        for clsname in htimage:
            #print (clsname.get_attribute('src'))
            #store the links. 
            self.image_links.append(clsname.get_attribute('src'))
            
        ##make list unique
        self.image_links = list(set(self.image_links))   



    def download_images(self):

        """ Method to download images and rename based on order in list
            Not many images on the website. Pages visited only have logos

            Args:
                get_path (Method)   : Get the global data store relative path and create it, if it does not exist
                imgdir (str)        : Define a subfolder images within the global relative data store directory

        
            Returns:
                Results in the images being store in the images folder which is relative to where the script is triggered. 
        
        """
        self.get_path()
        print (self.p)
        imgdir = Path(f"{self.p}\images")
        imgdir.mkdir(parents=True, exist_ok=True)    

        for i, img  in enumerate(self.image_links):
            ##store the images but rename using index.
            fimage = (img.split('/')[-1]).replace('.png','')       
            urllib.request.urlretrieve(img,(f"{imgdir}\{fimage}_{i}.png"))     
        
    def get_uuid(self):
        """ Using uuid v4 class to define unique tags for data extractions

            Args:
                uuid (str) : Holds the unique tag to be used for folder name in global data directory and for insertion in to dictionary
        
            Returns:
                uuid
                    Called within class or can be called external
        
        """
        ###store uuid for dcit and folder name
        self.uuid = str(uuid.uuid4())

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




def deepmine(URL):
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

    guardscap= GuardianScarper(URL)  
    guardscap.load_and_accept_cookies()

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
    
    ##writte the dictioary data into a json file
    with (p / 'data.json').open('w') as opened_file:          
          json.dump(guardscap.summary_det,opened_file)

    ##download the images
    guardscap.download_images()

    ##neatly close the selnium edge driver
    guardscap.driver.quit()
        


if __name__ == '__main__':
    ##set the global link for now
    URL = "https://www.mycancergenome.org/content/biomarkers/#search=TP53"
    deepmine(URL)