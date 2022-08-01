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

'''
Creates a basic scraper for  TP53 the guardian of the genome.  The selenium driver reads the 1st page and scrolls till it captures all the cards

Next steps are to merge uuid to self.summary_det



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
        '''
        Returns a list with all the links in the current page
        Parameters
        ----------
        driver: webdriver.Chrome
            The driver that contains information about the current page
        
        Returns
        -------
        link_list: list
            A list with all the links in the page
        '''
        ##has to be class directly parent of rest of listings
        ##needto be agnostic to nuance naming changes                                   cards small-up-1 medium-up-2 large-up-2 columns
        tp53_container = self.driver.find_element(by=By.XPATH, value="//div[contains(@class,'small-up-1 medium-up-2 large-up-2')]")
        #input[contains(@id,'id')]
        tp53_list = tp53_container.find_elements(by=By.XPATH, value='//div[@class="card-section"]')

        for tp53_property in tp53_list:
            a_tag = tp53_property.find_element(by=By.TAG_NAME, value='a')
            link = a_tag.get_attribute('href')
            #cardtitle = tp53_property.find_element(by=By.TAG_NAME, value='h5')
            #card_title = cardtitle.get_attribute('title')
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


            ###isssue >> tp53 only ever stays in the 1st car and refuses to cycle to next ???? 

            Redundant function.  Good learning though!
        '''


         ##has to be class directly parent of rest of listings
        tp53_container = tp53_property.find_element(by=By.XPATH, value='//*[@class="news-card-article"]')
        tp53_div = tp53_container.find_element(by=By.XPATH, value='./div')
        tp53_list = tp53_div.find_elements(by=By.XPATH, value='./p')
        ##tidy up naming
        tp53p_list = tp53_container.find_elements(by=By.XPATH, value='./p')
        summary_det = {}
        for tp53_prop in tp53_list:            
            ###get the key for this 1st div
            innertext = tp53_prop.text
            ###add to dictionary
            span_tag, inner_text = innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }           
            #print (inner_text)
            #print(span_tag.text)

        ###need to get the p tags now            
        for tp53p_prop in tp53p_list:
           
            #print ('p-body')       
            ###get the inner text encapsulated as a react text. 
            innertext= tp53p_prop.text
            ###add to dictionary
            span_tag, inner_text = innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }
                 
        #print(link_list)
        #time.sleep(30)

    def scroll_down(self):
        self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")


    def get_link_details(self, big_list):    

        for lnk in big_list:
            ##now in the card details i.e TP53 nonsense ...
            self.driver.get(lnk)
            ##in this case there is no images bar the logo and pathway need to traverse links to get them
            ###there is only one pathway map therefore do not store muliple times
            self.get_image()
            #print (lnk)
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

        self.get_path()
        print (self.p)
        imgdir = Path(f"{self.p}\images")
        imgdir.mkdir(parents=True, exist_ok=True)    

        for i, img  in enumerate(self.image_links):
          
            fimage = (img.split('/')[-1]).replace('.png','')

            print (img)
            urllib.request.urlretrieve(img,(f"{imgdir}\{fimage}_{i}.png"))     
        
    def get_uuid(self):
        self.uuid = str(uuid.uuid4())

    def get_path(self):

        self.p = Path(f"..\..\sandbox\DC_env\{self.summary_det['uuid']}")
        self.p.mkdir(parents=True, exist_ok=True)

        return self.p 







def deepmine(URL):

    ##get the json file ready



    guardscap= GuardianScarper(URL)  
    guardscap.load_and_accept_cookies()

    big_list = []
    for i in range(1): # The first 5 pages only
        #I am only human
        
        big_list.extend(guardscap.get_links()) # Call the function we just created and extend the big list with the returned list
        
        #print (big_list)
        print(f'There are {len(big_list)} tp53 groups so far')
        #nextpage = driver.find_element(by=By.XPATH, value='//div[@class="scroller-button"]')
        #nextpage.click()
        time.sleep(3)

    ##pack the list back to guardian let it handle it
    guardscap.get_link_details(big_list)


    ##dump this data to v4 uuid folder
    p = guardscap.get_path()  
    
    with (p / 'data.json').open('w') as opened_file:          
          json.dump(guardscap.summary_det,opened_file)


    ##download the images
    guardscap.download_images()



    guardscap.driver.quit()
        
'''

    for tp53_link in big_list:
        
        layout_container = tp53_link.find_element(by=By.XPATH, value='//div[@class=small-12 columns"]')    
        print(layout_container)
        plan = layout_container.find_elements(by=By.XPATH, value='./div')

        for house_plan in plan:
            print (house_plan)
            try:
                p_tag = house_plan.find_element(by=By.TAG_NAME, value='p')
                attribu = p_tag.get_attribute('data-testid')
                #print (attribu, val)
                #print ("deal")       
                ##     
                val = p_tag.find_element(by=By.XPATH,value="//*[@data-testid='" + attribu +"']").text
                #price = p_tag.find_element(by=By.XPATH,value='//*[@data-testid="price"]').text
                print (attribu,'<<<<>>>>>', val)
                #print (val)
            except:
                ##some div do not have p tag therefore errors out
                continue   
            print (p_tag)  
                
'''

        


if __name__ == '__main__':
    URL = "https://www.mycancergenome.org/content/biomarkers/#search=TP53"
    deepmine(URL)