from pymongo import MongoClient

class MongoDBService:
    def __init__(self, database_name:str, collection_name:str):
        # Connect to the MongoDB server (adjust the URI if necessary)
        self.client = MongoClient("mongodb://localhost:27017/")

        # Select the desired database (it will be created if it doesn't exist)
        self.db = self.client[database_name]

        # Select the desired collection (it will be created if it doesn't exist)
        self.collection = self.db[collection_name]

    def insert_document(self, document_dict):
        result_insert = self.collection.insert_one(document_dict)

        # Display the _id of the inserted document
        print("Document inserted with _id:", result_insert.inserted_id)