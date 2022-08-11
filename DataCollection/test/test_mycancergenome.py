from MyCancerGenome import mycancergenome_scraper
import unittest

'''
Unit test for mycancergenome mining software. 

#TODO Regression testing 


'''

class MyCancerGenomeTestCases(unittest.TestCase):
      
    def test_get_links(self):
          ###short hand notation
        mcs = mycancergenome_scraper.GuardianScarper()
        mcs.WS.load_and_accept_cookies()
        set_of_links = mcs.get_links('small-up-1 medium-up-2 large-up-2 offset')  
       

        self.assertGreater(len(set_of_links),0)
        self.assertTrue(type(set_of_links) == list)


    def test_link_details(self):
      
        mcs = mycancergenome_scraper.GuardianScarper()
        mcs.WS.load_and_accept_cookies()
        tlink =['https://www.mycancergenome.org/content/alteration/tp53-mutation/' , 'https://www.mycancergenome.org/content/alteration/tp53-mutation/'] 
        mcs.get_link_details(tlink)
        ###test that we have a uniq list still
        print (mcs.clintrials_link)
        self.assertEqual(len(mcs.clintrials_link),len(set(mcs.clintrials_link)))
        self.assertTrue(type(mcs.clintrials_link) == list)

    def test_get_clinical_trial_details(self):
        mcs = mycancergenome_scraper.GuardianScarper()
        mcs.WS.load_and_accept_cookies()
        mcs.get_clinical_trial_details()

        self.assertTrue(type(mcs.summary_det) == dict)
        ##standard  key should be retuned for default link
        self.assertIn('AflacLL1901 (CHOA-AML)',mcs.summary_det.keys())
        

unittest.main(argv=[''], verbosity=2, exit=False)