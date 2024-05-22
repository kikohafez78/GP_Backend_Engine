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

groups = Blueprint('groups', __name__)

@groups.route("/",methods = ["POST"])
def create_group():
    data = request.get_json()
    admin = data["_id"]
    user = None
    try:
        user = mydb.User.find_one({"_id": ObjectId(admin)})
        if user == None:
            return jsonify({"message": "admin user is not found"}), 404
    except:
        return jsonify({"message": "admin id is not valid"}), 400
    group_name = data["title"]
    if mydb.Groups.find_one({"group_name": group_name}) is not None:
        return {"message" : "a group with that name already exists"}, 401
    group_description = data.get("description","")
    group_privacy_settings = data.get("privacy_type", "public")
    password = ""
    if group_privacy_settings.lower() != "public":
        password = data.get("password","")
    member_ids = data.get("member_ids", None)
    members = []
    if member_ids is not None:
        for member_id in member_ids:
            try:
                member = mydb.User.find_one({"_id": ObjectId(member_id)})
                if member != None:
                    members.append(ObjectId(member_id))
            except:
                pass
    group_id = mydb.Groups.insert_one({
        "group_name": group_name,
        "description": group_description,
        "admin_ids": [admin],
        "member_ids": member_ids,
        "messages": [],
        "media":[],
        "sheet_repositories":[],
        "member_notes": {ObjectId(member_id):[] for member_id in member_ids},
        "password": password
    })
    if group_id is not None:
        return {"message": f"group id is {group_id}"}, 200
    else:
        return {"message": "service is unavailable, please try again later"}, 404
            
    
@groups.route("/group_id", methods = ["PUT"])
def add_user():
    data = request.get_json()
    admin = data["_id"]
    try:
        user = mydb.User.find_one({"_id": ObjectId(admin)})
        if user == None:
            return jsonify({"message": "admin user is not found"}), 404
    except:
        return jsonify({"message": "admin id is not valid"}), 400
    group_id = data.get("group_id", None)
    group_name = data.get("group_name", None)
    group = None
    if group_id is None and group_name is None:
        return jsonify({"message": "the provided information is missing both group name and group id"}), 400
    elif group_name is not None:
        group = mydb.Groups.find_one({"group_name": group_name})
    elif group_id is not None:        
        try:
            group = mydb.User.find_one({"_id": ObjectId(group_id)})
            if group == None:
                return jsonify({"message": "group name is not found"}), 404
        except:
            return jsonify({"message": "admin id is not valid"}), 400
    if group["admin_id"] != admin:
        return {"message" : "the requesting user is not an admin of the group"}, 401
    new_member_id = data.get("member_id",None)
    try:
        user = mydb.User.find_one({"_id": ObjectId(new_member_id)})
        if user == None:
            return jsonify({"message": "new memeber user is not found"}), 404
    except:
        return jsonify({"message": "member id is not valid"}), 400
    group["member_ids"].append(new_member_id)
    group["member_notes"][ObjectId(new_member_id)] = []
    if mydb.Groups.update_one(group,{"_id" : ObjectId(group_id)}) is not None:
        return jsonify({"message": "memebr with id {new_member_id} is added successfully!!"}), 200
    else:
        return jsonify({"message" : "the request was unsuccessfull, please try again later"}), 405