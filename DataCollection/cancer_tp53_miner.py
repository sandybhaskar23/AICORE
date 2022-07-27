from turtle import title
from attr import attrib
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import time

'''
driver = webdriver.Edge() 
URL = "https://www.zoopla.co.uk/new-homes/property/london/?q=London&results_sort=newest_listings&search_source=new-homes&page_size=25&pn=1&view_type=list"
driver.get(URL)
time.sleep(2) # Wait a couple of seconds, so the website doesn't suspect you are a bot
try:
    driver.switch_to_frame('gdpr-consent-notice') # This is the id of the frame
    accept_cookies_button = driver.find_element(by=By.XPATH, value='//*[@id="save"]')
    accept_cookies_button.click()

except AttributeError: # If you have the latest version of Selenium, the code above won't run because the "switch_to_frame" is deprecated
    driver.switch_to.frame('gdpr-consent-notice') # This is the id of the frame
    accept_cookies_button = driver.find_element(by=By.XPATH, value='//*[@id="save"]')
    accept_cookies_button.click()

except:
    pass
time.sleep(2)
dict_properties = {'Link' : [] , 'Price': [], 'Address': [], 'Bedrooms': []}
house_property = driver.find_element(by=By.XPATH, value='//*[@id="listing_62009695"]') # Change this xpath with the xpath the current page has in their properties

a_tag = house_property.find_element(by=By.TAG_NAME, value='a')

link = a_tag.get_attribute('href')
print(link)
dict_properties['link'].append(link)
price = house_property.find_element(by=By.XPATH, value='//*[@data-testid="listing-price"]').text
print(price)
dict_properties['Price'].append(price)
address = house_property.find_element(by=By.XPATH, value='//*[@data-testid="listing-description"]').text
print(address)
dict_properties['Address'].append(address)
bedrooms = house_property.find_element(by=By.XPATH, value='//p[@class="css-r8a2xt-Text eczcs4p0"]').text
print(bedrooms)
dict_properties['Bedrooms'].append(bedrooms)
#div_tag = house_property.find_element(by=By.XPATH, value='//div[@data-testid="truncated_text_container"]')
#does not exist or equlivent even ->> span_tag = div_tag.find_element(by=By.XPATH, value='.//span')
#description = span_tag.text
#print(description)
'''

class GuardianScarper:


    def __init__(self,URL):
        self.summary_det = {}
        self.link_list =[]


    def load_and_accept_cookies(self):
        
        #Open Zoopla and accept the cookies
        
        #Returns
        ##-------
        #driver: webdriver.Chrome
        #   This driver is already in the Zoopla webpage
        
        driver = webdriver.Edge() 
        
        driver.get(URL)
        time.sleep(3) 
        ##get driver to wait 10 seconds
        delay = 10
        try:
            WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.XPATH, '//*[@id="gdpr-consent-notice"]')))
            print("Frame Ready!")
            driver.switch_to.frame('gdpr-consent-notice')
            accept_cookies_button = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.XPATH, '//*[@id="save"]')))
            print("Accept Cookies Button Ready!")
            accept_cookies_button.click()
            time.sleep(1)
        except TimeoutException:
            print("Loading took too much time! to be expected no cookes at this website")
            ###there are no cookies consent request for this website
            pass
        except:
            pass

        return driver  # If there is no cookies button, we won't find it, so we can pass
    #element = driver.find_element(By.ID, 'sb_form_q')
    #element.send_keys('WebDriver')
    #element.submit()

    def get_links(self, driver):
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
        tp53_container = driver.find_element(by=By.XPATH, value='//div[@class="small-up-1 medium-up-2 large-up-2 offset-2 columns"]')

        
        tp53_list = tp53_container.find_elements(by=By.XPATH, value='//div[@class="card-section"]')

        for tp53_property in tp53_list:
            a_tag = tp53_property.find_element(by=By.TAG_NAME, value='a')
            link = a_tag.get_attribute('href')
            cardtitle = tp53_property.find_element(by=By.TAG_NAME, value='h5')
            card_title = cardtitle.get_attribute('title')
            print ('start')
            print (link)
            self.link_list.append(link)
            chunks = tp53_property.text.split('\n')
                
            self.summary_det[chunks[0]] ={'link':link}
            ###convert rest in list into dictionary
            res_dct = {chunks[i].split(':')[0]: chunks[i].split(':')[-1] for i in range(1, len(chunks))}
            
            self.summary_det[chunks[0]].update(res_dct)
            print(self.summary_det)
            
            print (chunks)
            #span_tag = tp53_property.find_element(by=By.XPATH, value='//span[@class="title-case"]')
            ###get the inner text encapsulated as a react text. 
           
            ###tidy up this function it does not parse the data efficiently 
            #self.get_summary(tp53_property,card_title)
            
        print(f'There are {len(self.link_list)} properties in this page')
        
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
    
        for tp53_prop in tp53_list:

            
            ###get the key for this 1st div
            innertext = tp53_prop.text
            ###add to dictionary
            span_tag, inner_text = innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }
           # self.summary_det[card_title] = {span_tag : inner_text }
            #print (inner_text)
            #print(span_tag.text)

        ###need to get the p tags now            
        for tp53p_prop in tp53p_list:
           
            print ('p-body')
            ###get the key for this 1st div
            #spanp_tag = tp53p_property.find_element(by=By.XPATH, value='//span[@class="title-case"]')
        
            ###get the inner text encapsulated as a react text. 
            #inner_text = tp53p_property.find_element(by=By.XPATH, value='./p').text
            innertext= tp53p_prop.text
            ###add to dictionary
            span_tag, inner_text = innertext.split(':')    
            self.summary_det[card_title] = {span_tag : inner_text }
            print (inner_text)
            #print(spanp_tag.text)
            print('end p-body')

        tp53_property = None
        
        #print(link_list)
        #time.sleep(30)
    

def deepmine(URL):

    guardscap= GuardianScarper(URL)

  
    driver = guardscap.load_and_accept_cookies()
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    big_list = []
    for i in range(1): # The first 5 pages only
        #I am only human
        
        big_list.extend(guardscap.get_links(driver)) # Call the function we just created and extend the big list with the returned list
        ## TODO: Click the next button. Don't forget to use sleeps, so the website doesn't suspect
        #print (big_list)
        print(f'There are {len(big_list)} tp53 so far')
        #nextpage = driver.find_element(by=By.XPATH, value='//div[@class="scroller-button"]')
        #nextpage.click()
        time.sleep(3)

        


    for prop_link in big_list:
        ## TODO: Visit all the links, and extract the data. Don't forget to use sleeps, so the website doesn't suspect
        #pass # This pass should be removed once the code is complete
        #for prop_link in big_list:
        time.sleep(10)
        print (prop_link)
        driver.get(prop_link)
        ##make sure only one element otherwise returns list with no objects only
        ##this is in each listing and requires daily updates.  the css name changes routinely
        #layout_container = driver.find_element(by=By.XPATH, value='//*[@class="c-PJLV c-PJLV-ideupcC-css"]')
        layout_container = driver.find_element(by=By.XPATH, value='//div[@data-testid="listing-summary-details"]')    
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
                


    driver.quit()


if __name__ == '__main__':
    URL = "https://www.mycancergenome.org/content/biomarkers/#search=TP53"
    deepmine(URL)