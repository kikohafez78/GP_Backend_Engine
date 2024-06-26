from flask_restful import Resource, Api, reqparse
from flask import Flask, request, Response, jsonify, Blueprint
from flask_cors import cross_origin
from bson.objectid import ObjectId
import jwt
import json
from functools import wraps
from Database.Database import Database as mydb
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# try:
#     app = Flask(__name__)
#     client = pymongo.MongoClient(
#         "mongodb+srv://karimhafez:KojGCyxxTJXTYKYV@cluster0.buuqk.mongodb.net/admin")
#     app.db = client.Twitter_new
# except:
#     print("can't connect")

# api = Api(app)

get_user = Blueprint('get_user', __name__)


############################################

@get_user.route("/user_id", methods=["GET"])
def singleuser():
    query = {"_id": ObjectId(request.args["_id"])}
    user = mydb.User.find_one(query)
    if user == None:
        return Response(
            response=json.dumps(
                {"message": "User ID not found"
                }),
            status=404,
            mimetype="application/json"
        )
    else:
        db_response = mydb.User.find_one(query)
        if 'password' in db_response:
            del db_response['password']
        if 'notifications' in user:
            del db_response['notifications']
        db_response["creation_date"] = db_response["creation_date"].date()
        db_response["creation_date"] = db_response["creation_date"].strftime("%Y-%m-%d")
        db_response["_id"] = str(db_response["_id"])
        # print(db_response)
        return Response(
            response=json.dumps(
                {"message": "The request was succesful",
                "user": db_response
                }),
            status=200,
            mimetype="application/json")

   

