import bcrypt
from flask import Blueprint, request, jsonify, request
import jwt
from Database.Database import Database as mydb
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)




Login = Blueprint("Login" ,__name__)




@Login.route("/", methods=['POST'])
def Home():
    email = request.form.get('email')
    password = request.form.get('password')
    if password == None:
        password = "NULL"
    password_byte = bytes(password, "ascii")
    
    isfound = mydb.User.find_one({'email': email})

    if isfound and password == "NULL":
        return jsonify({"message": "user is found but the password is wrong"}), 203
    elif isfound:
        if bcrypt.checkpw(password_byte, isfound["password"]):
            del isfound["password"]
            isfound["_id"] = str(isfound["_id"])
            return jsonify({'message': "user found", "user": isfound }), 200
        else:
            return jsonify({"message": "email or password may not be incorrect"}), 400
    else:
            return jsonify({"message": "user doesn't exist"}),404




    







       

    


    



  
    
