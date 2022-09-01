from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import time
import uuid


class WebScraper:

    def __init__(self,driver):
        self.driver = driver
        


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

    def scroll_down(self):
        """ Ensures HTML is fully loaded to  allow the whole page to be mined

            Args:
                driver: webdriver.Chrome scrolldown function
            
            Returns:
                Results in the webpage being fully loaded
                    
        """
        ##make sure the html is fully loaded
        self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(1)

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

    def get_image(self,driver):

        """ Extracts all the links from the href tags

            Args:
                driver              : webdriver.Chrome get function for link
                
        
            Returns:
                image_links 
                    Contains all the unique links found on the web pages loaded
        
        """
        ##logo
        ##html class name. 
        #htclass_name = ['nav-bar show-for-medium','associated-pathways']
        htimage = driver.find_elements(by=By.TAG_NAME, value = 'img')
        image_links = []
        for clsname in htimage:
            #print (clsname.get_attribute('src'))
            #store the links. 
            image_links.append(clsname.get_attribute('src'))
            
        ##make list unique
        image_links = list(set(image_links)) 

        return image_links 

