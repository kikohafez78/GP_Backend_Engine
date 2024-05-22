from pymongo import MongoClient
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
try:
    client = MongoClient("mongodb+srv://karimhafez37:kikohafez32@cluster0.co2vcwp.mongodb.net/admin")
    Database = client.Auto
    
except: 
    print("Error: Cannot connect to Database")

