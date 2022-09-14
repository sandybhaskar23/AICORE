from asyncio.windows_events import NULL
from importlib.metadata import metadata
from typing import Dict
from uuid import UUID, uuid5
from sqlalchemy import *
from sqlalchemy.schema import MetaData
from sqlalchemy.dialects.postgresql import UUID
from mcgconnection import TpConnection
import pandas as pd




class McsInterface:

    def __init__(self,data:dict=None, subheader:list=None,slice=None):

        self.db ,self.tables = TpConnection().db_connection()
        self.meta = MetaData()
        self.data ={}
        self.data = data
        self.subheader = subheader
        ty = type(self.data)
        print(ty)
        self.slice = slice



    def run(self):

        self.check_table_exist()
        self.create_table()
        self.write_to_tables()    

    def check_table_exist(self, table = None) -> None:

        if table not in self.tables:
            return None           
        
        

    def create_table(self)-> None:
        '''
       

        '''
        check = self.check_table_exist('Biomarkers')
        check_t2 = self.check_table_exist('ClinicalDetails')

        if check is None and check_t2 is None :
        
                ###write table
                biomarkers = Table(
                    'Biomarkers',self.meta,
                    Column('index', INTEGER,nullable=True),
                    Column('Id', UUID(as_uuid=True), primary_key=True), 
                    Column('Link',VARCHAR(100), nullable = False),
                    Column('Biomarkers',VARCHAR(30),nullable=False),                                     
                    Column('Diseases',VARCHAR(7000),nullable = True), 
                    Column('Clinical_Trials',VARCHAR(4),nullable =True),
                    Column('Drugs',VARCHAR(4),nullable =True),
                    Column('Alteration_Groups',VARCHAR(300),nullable=True),
                            )

            
                clinical_details = Table(
                    'ClinicalDetails',self.meta,
                    Column('index', INTEGER,nullable=True),
                    Column('Det_id',UUID(as_uuid=True), primary_key = True),
                    Column('Id',UUID(as_uuid=True),ForeignKey("Biomarkers.Id"),nullable = False), 
                    Column('Link', VARCHAR(100),nullable =True),
                    Column('Description',VARCHAR(8000),nullable = False),                               
                    Column('NCT_ID',VARCHAR(15),nullable =True),
                    Column('Phase',VARCHAR(40),nullable = True),
                    Column('Biomarkers',VARCHAR(1000),nullable = True),
                    Column('Diseases',VARCHAR(1000),nullable =True) 
                    )
                
                self.meta.create_all(self.db)
    

    def check_if_data_exists_by_primary(self,tble,col,val):
        self.no_data = 1
        stmt = f'select "{col}" from "{tble}" where "{col}" = \'{val}\' '
       
        results = self.db.execute(stmt).fetchall()

        if results is None:
            self.no_data = None

        return self.no_data

        

    def write_to_tables(self) -> None:

        '''
        Use panda to sql but need to first deal with columns
        Pandas cannot resolve complex data structures
        '''

        ##data is already a dataframe from 
        df = pd.DataFrame(self.data).transpose()   
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'Id'})
  
        # only one group has pathways and is is uninformative 
        df.pop('Pathways')
        df.pop(self.slice)
  
        #fill empty column in place with 0 since the db does tno like blank values
        df.fillna(0, inplace=True, downcast='infer')   
        #seed the data      
        df.to_sql('Biomarkers',self.db,if_exists='append',index=False)
        print('Nearly<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>')


        ###now write subtable 
        self.subtable=[]
        self.subtable.append(self.subheader)
        
        for unique_id in self.data:
            try :
                for k_id in self.data[unique_id][self.slice]:
                    self.subtable.append(k_id)   
            except  KeyError:
                print(f'{unique_id} does not have {self.slice} key')
        
        dfs = pd.DataFrame(self.subtable)     
        dfs.columns = dfs.iloc[0]
        #remove first row from DataFrame
        dfs = dfs[1:]
     
        #insert second table
        dfs.to_sql('ClinicalDetails',self.db,if_exists='append')       
 
        print('<<<<<<<<<<<<<<<Done-Seeding>>>>>>>>>>>>>>>>>>>>>>>>>>>')
   