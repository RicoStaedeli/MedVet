import sqlite3
import datetime
from Utils.logger import get_logger

class DbWriter:
    
    def __init__(self,config: dict):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.logger.debug(f"Connect to Database")
        conn = sqlite3.connect('Database/medvet_chat_db.sqlite')
        cur = conn.cursor()
        #cur.execute('DROP TABLE IF EXISTS chatHistory')
        self.logger.debug(f"Create Table if not exists")
        cur.execute('CREATE TABLE IF NOT EXISTS chatHistory (id_chat INTEGER PRIMARY KEY,creationDate timestamp, agent_id VARCHAR, prompt_user TEXT NOT NULL, answer_assistant TEXT NOT NULL)')
        #cur.execute('DELETE * FROM chatHistory')
        conn.commit()
        self.logger.debug(f"Close connection to Database")
        conn.close()

    def insert_chat(self,prompt_user,agent_id,answer_assistant ):
        try:
            connection = sqlite3.connect('Database/medvet_chat_db.sqlite')
            cur = connection.cursor()
            sqlite_insert_query = '''INSERT INTO chatHistory (creationDate, agent_id, prompt_user, answer_assistant) VALUES (?, ?, ?, ?)'''
            creationDate = datetime.datetime.now()
            cur.execute(sqlite_insert_query,(creationDate, agent_id, prompt_user, answer_assistant))
            #id_chat = cur.lastrowid
            connection.commit()
            connection.close()
            self.logger.debug(f'Insert execution successfull')
            return True
        except Exception as e:
            self.logger.error(f'Error executing insert query: {e}')
            if connection:
                connection.close()
        return False

    def find(self,id_chat):
        connection = sqlite3.connect('Database/medvet_chat_db.sqlite')
        cur = connection.cursor()
        sqlite_select_query = f"""SELECT * FROM chatHistory WHERE id_chat={id_chat}"""
        self.logger.debug(f'Execute query: {sqlite_select_query}')
        cur.execute(sqlite_select_query)
        entry = cur.fetchall()
        connection.commit()
        connection.close()
        return entry
    
    def find_by_agentId(self,agent_id):
        connection = sqlite3.connect('Database/medvet_chat_db.sqlite')
        cur = connection.cursor()
        sqlite_select_query = f"""SELECT * FROM chatHistory WHERE agent_id='{agent_id}'"""
        self.logger.debug(f'Execute query: {sqlite_select_query}')
        cur.execute(sqlite_select_query)
        entries = cur.fetchall()
        connection.commit()
        connection.close()
        return entries
    
    