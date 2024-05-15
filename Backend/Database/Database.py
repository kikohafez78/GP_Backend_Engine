from pymongo import MongoClient
try:
    client = MongoClient("mongodb+srv://karimhafez:KojGCyxxTJXTYKYV@cluster0.buuqk.mongodb.net/admin")
    Database = client.Auto
except: 
    print("Error: Cannot connect to Database")

