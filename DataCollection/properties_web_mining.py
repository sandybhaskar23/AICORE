from selenium import webdriver
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




def load_and_accept_cookies():
    
    #Open Zoopla and accept the cookies
    
    #Returns
    ##-------
    #driver: webdriver.Chrome
     #   This driver is already in the Zoopla webpage
    
    driver = webdriver.Edge() 
    URL = "https://www.zoopla.co.uk/new-homes/property/london/?q=London&results_sort=newest_listings&search_source=new-homes&page_size=25&pn=1&view_type=list"
    driver.get(URL)
    time.sleep(3) 
    try:
        driver.switch_to_frame('gdpr-consent-notice') # This is the id of the frame
        accept_cookies_button = driver.find_element(by=By.XPATH, value='//*[@id="save"]')
        accept_cookies_button.click()
        time.sleep(1)
    except AttributeError: # If you have the latest version of Selenium, the code above won't run because the "switch_to_frame" is deprecated
        driver.switch_to.frame('gdpr-consent-notice') # This is the id of the frame
        accept_cookies_button = driver.find_element(by=By.XPATH, value='//*[@id="save"]')
        accept_cookies_button.click()
        time.sleep(1)

    except:
        pass

    return driver  # If there is no cookies button, we won't find it, so we can pass
#element = driver.find_element(By.ID, 'sb_form_q')
#element.send_keys('WebDriver')
#element.submit()

def get_links(driver: webdriver.Edge) -> list:
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
    prop_container = driver.find_element(by=By.XPATH, value='//div[@class="css-1itfubx ejttno50"]')

    prop_list = prop_container.find_elements(by=By.XPATH, value='./div')
    link_list = []
    #<div class="css-1cbtsvd e1xegeql28"><
    for house_property in prop_list:
        a_tag = house_property.find_element(by=By.TAG_NAME, value='a')
        link = a_tag.get_attribute('href')
        link_list.append(link)
        
    print(f'There are {len(link_list)} properties in this page')
    #print(link_list)
    #time.sleep(30)
    return link_list

big_list = []
driver = load_and_accept_cookies()

for i in range(2): # The first 5 pages only
    #I am only human
    
    big_list.extend(get_links(driver)) # Call the function we just created and extend the big list with the returned list
    ## TODO: Click the next button. Don't forget to use sleeps, so the website doesn't suspect
    #print (big_list)
    print(f'There are {len(big_list)} properties so far')
    nextpage = driver.find_element(by=By.XPATH, value='//a[@class="eaoxhri5 css-xtzp5a-ButtonLink-Button-StyledPaginationLink eaqu47p1"]')
    nextpage.click()
    time.sleep(3)

    


for prop_link in big_list:
    ## TODO: Visit all the links, and extract the data. Don't forget to use sleeps, so the website doesn't suspect
    #pass # This pass should be removed once the code is complete
    #for prop_link in big_list:
    time.sleep(2)
    print (prop_link)
    driver.get(prop_link)
    ##make sure only one element otherwise returns list with no objects only
    layout_container = driver.find_element(by=By.XPATH, value='//*[@class=c-PJLV c-PJLV-iiskFxm-css"]')
    print(layout_container)
    plan = layout_container.find_elements(by=By.XPATH, value='./div')
#c-PJLV c-PJLV-iiskFxm-css
    for house_plan in plan:
        #print (house_plan)
     
        p_tag = house_plan.find_element(by=By.TAG_NAME, value='p')
        
            


driver.quit()
