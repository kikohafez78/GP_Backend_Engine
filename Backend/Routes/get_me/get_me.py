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

get_me = Blueprint('me', __name__)



@get_me.route("/", methods=["GET"])
def me():
    user_name = request.form.get("user_name")
    user = mydb.User.find_one({"user_name": user_name})
    if user is None:
        return {"message" : "user is not found in the database"}, 404
    user["_id"] = str(user["_id"])
    return Response(response=json.dumps({"message": "user is found", "user": user}),status=200,mimetype='application/json')
    
   

@get_me.route("/sessions", methods = ["GET"])
def my_sessions():
    user_name = request.form.get("user_name")
    user = mydb.User.find_one({"user_name": user_name})
    if user is None:
        return {"message" : "user is not found in the database"}, 404
    
    user["_id"] = str(user["_id"])
    return Response(response=json.dumps({"message": "user is found", "user": user}),status=200,mimetype='application/json')