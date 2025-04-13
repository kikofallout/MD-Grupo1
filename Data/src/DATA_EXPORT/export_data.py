import pymongo
from pymongo import MongoClient
import json

# Conectar ao MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["nutrition"]
collection = db["data"]

# Exportar para JSON
cursor = collection.find()
with open("saida.json", "w") as file:
    for document in cursor:
        file.write(json.dumps(document, default=str) + "\n")
