import datetime
from google.cloud import bigquery
from scripts.log import logger


class Session:
    def __init__(self):
        # Initialize BigQuery client
        try:
            self.client = bigquery.Client()
        except Exception as e:
            self.client = None
            logger.error(str(e))
        # Define constants for BigQuery
        self.PROJECT_ID = "hclsw-gcp-xai"
        self.DATASET_ID = "voltmx_chatbot"
        self.TABLE_ID = 'chat_history'
        self.TABLE_REF = f'{self.PROJECT_ID}.{self.DATASET_ID}.{self.TABLE_ID}'

    def is_conversation_exist(self, session_id):
        query = f"""
                SELECT * FROM `{self.PROJECT_ID}.{self.DATASET_ID}.{self.TABLE_ID}` where session_id = "{session_id}" 
                LIMIT 1
                """
        query_job = self.client.query(query)
        results = query_job.result()
        return len([row for row in results]) != 0

    def get_conversations(self, session_id, memory_window):
        query = f"""
                SELECT * FROM `{self.PROJECT_ID}.{self.DATASET_ID}.{self.TABLE_ID}` where session_id = "{session_id}" 
                ORDER BY created_on DESC LIMIT {memory_window}
                """
        query_job = self.client.query(query)
        results = query_job.result()
        return results

    def add_to_conversations(self, message_id, session_id, message=None, response=None):
        rows_to_insert = [
            {

                "created_on": datetime.datetime.utcnow().isoformat(),
                "message_id": message_id,
                "session_id": session_id,
                "message": message,
                "response": response
            }
        ]
        errors = self.client.insert_rows_json(self.TABLE_REF, rows_to_insert)
        if errors:
            raise Exception(f"Encountered errors while inserting rows: {errors}")
        return session_id

