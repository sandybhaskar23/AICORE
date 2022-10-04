from mcgconnection import TpConnection
from sqlalchemy import Column, Table , VARCHAR, INTEGER , ForeignKey
from sqlalchemy.schema import MetaData
from sqlalchemy.dialects.postgresql import UUID
import pandas as pd




class McsInterface:

    def __init__(self,data:dict=None, subheader:list=None,slice=None):

        self.db ,self.tables = TpConnection().db_connection()
        self.meta = MetaData()
        self.data ={}
        self.data = data
        self.subheader = subheader
        ty = type(self.data)        
        self.slice = slice



    def run(self):
        
        """ Runs McsInterface methods in below order:
            check_table_exis
            _create_table
            write_to_tables

            Args: None except what is past to the class instantiation 
            
           Creates two tables in database 
            
            Returns: 
                    Nothing                
        
        """        
        self._create_table()
        self.write_to_tables()    

    def check_table_exist(self, table = None) -> None:
        """Checks if a table exist

        Args: Table name

        Returns:
                None if no table exist     

        """       
        if table in self.tables:            
            return table
        else:
            return None
                   
        
        

    def _create_table(self)-> None:
        '''Uses sqlalchemy meta engine to create table model and then create them

        Args: 
            No args given

        Return:
            Checks if table Biomarkers and ClinicalDetails exist if not creates models and table in AWS system

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
    

    def check_if_data_exists_by_primary(self,tble,col=None,val=None):
        """Checks if data in a table exist

        Args:
            tble = Table name
            col = table column 
            val = the data being chedcked

        Returns:
            'None' if no data exists
            '1' if data exists
    
        """
        check = self.check_table_exist(tble)
        
        ##1st check for tables existence 
        if check is None:
            return check
       
        ##table exists not make sure col has a value
        ##initialise return value 
        self.no_data = None
        if col is not None:
           
            stmt = f'select "{col}" from "{tble}" where "{col}" = \'{val}\' '
        
            results = self.db.execute(stmt).fetchall()
            print (f"Data exist {results}")
            if results is not None:
                self.no_data = results
        
        return self.no_data

        

    def write_to_tables(self) -> None:

        '''
        Uses pandas to insert data into SQL.Flatten data for subtable before insert
        Pandas cannot resolve complex data structures

        Args:
            None given uses args passed to class at instantiation
        Returns:
            Nothing

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
   