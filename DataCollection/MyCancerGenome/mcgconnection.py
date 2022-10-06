import db_config
import psycopg2
from sqlalchemy import create_engine
import os
from os.path import join, dirname
from pathlib import Path
from  dotenv import load_dotenv 



class TpConnection:
    def __init__(self) -> None:
          self.conn = db_config.config()
          dotenv_path = join(dirname(__file__), '.env')          
          load_dotenv(dotenv_path)
     

    def db_connection(self):         
            self.engine = create_engine(f"postgresql+psycopg2://{os.getenv('USRNAME')}:{os.getenv('PASSWORD')}@{os.getenv('HOST')}/{os.getenv('DATABASE')}")
            self.tables = self.engine.table_names()
          
            return self.engine,self.tables  