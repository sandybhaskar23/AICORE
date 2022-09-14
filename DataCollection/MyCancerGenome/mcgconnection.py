import db_config
import psycopg2
from sqlalchemy import create_engine


class TpConnection:
    def __init__(self) -> None:
          self.conn = db_config.config()

    def db_connection(self):
            self.engine = create_engine(f"postgresql+psycopg2://{self.conn['USERNAME']}:{self.conn['PASSWORD']}@{self.conn['HOST']}/{self.conn['DATABASE']}")
            self.tables = self.engine.table_names()
          
            return self.engine,self.tables  