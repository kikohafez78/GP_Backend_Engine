from flask_restful import Resource, Api, reqparse
from flask import Flask, request, Response, jsonify, Blueprint
from flask_cors import cross_origin
from bson.objectid import ObjectId
import jwt
import json
from functools import wraps
from Database.Database import Database as mydb
from app import Engine
import datetime

chat = Blueprint('chat', __name__)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, "SecretKey1911", "HS256")
            user_id = ObjectId(data['_id'])
            current_user = mydb.User.find_one({'_id': user_id})

        except:
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(current_user, *args, **kwargs)

    return decorated



@chat.route("/user_id", methods=["POST"])
@token_required
def create_chat(current_user):
    query = {"_id": ObjectId(request.args["_id"])}
    user = mydb.User.find_one(query)
    if user == None:
        return Response(
            response=json.dumps(
                {"message": "User ID not found then no chat can be created"
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
        del db_response["creation_date"].date()
        db_response["_id"] = str(db_response["_id"])
        db_response["chat"] = {"head_message":None,"messages":[]}
        try:
            chat_data = {
                "Title": request.args["Title"],
                "messages":{},
                "creation_date": datetime.date.ctime()  
            }
            mydb.Chat.insert_one(chat_data)
            
        except:
            return Response(
            response=json.dumps(
                {"message": "could not create chat at the moment, please try again later"
                }),
            status=405,
            mimetype="application/json"
        )
        # print(db_response)
        return Response(
            response=json.dumps(
                {"message": "The request was succesful",
                 "user": chat_data
                }),
            status=200,
            mimetype="application/json")
        
        
@chat.route("/user_id", methods=["GET"])
def get_chat():
    query = {"_id": ObjectId(request.args["_id"])}
    user = mydb.User.find_one(query)
    if user == None:
        return Response(
            response=json.dumps(
                {"message": "User ID not found then no chat can be retrieved"
                }),
            status=404,
            mimetype="application/json"
        )
    else:
        pass