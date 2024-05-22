from urllib import response

from pyparsing import empty
from flask import Blueprint, request, Response, jsonify, render_template
from pymongo import MongoClient
from flask_cors import cross_origin
import jwt
from bson import ObjectId
from functools import wraps
from Database.Database import Database 
import re
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)



GetAll = Blueprint("GetAll" ,__name__)
    




@GetAll.route("/all", methods=['GET'])
def GET_ALL():
    
    if request.args['admin'] == True:
        count = Database.User.count_documents({})
        limit = int(request.args.get('limit'))
        offset = int(request.args.get('offset'))
        empty_array = []
        empty_document = {}
        empty_array.append(empty_document)
        if (offset > count - 1):
            return jsonify({"users": empty_array}), 204

        starting_id = Database.User.find().sort('_id')
        last_id = starting_id[int(offset)]['_id']
        isfound = Database.User.find({'_id': {'$gte': last_id}}).sort('_id').limit(limit)
        output = []

    

        for i in isfound:
            i["_id"] = str(i["_id"])
            i["creation_date"] = i["creation_date"].date()
            i["creation_date"] = i["creation_date"].strftime("%Y-%m-%d")
            if 'password' in i:
                del i['password']
            if 'notifications' in i:
                del i['notifications']
            
            output.append(i)
        
        return jsonify({"users": output}), 200
        
    else: 
        return jsonify({"Message": "user is not admin"}), 403

   



@GetAll.route("/search", methods=['GET'])
# @cross_origin(allow_headers=['Content-Type', 'x-access-token', 'Authorization'])
def search_user():
    blockers = []
    blocking = []
    paginated_list = []
    empty_array = []
    users = []
    limit = int(request.args.get('limit'))
    offset = int(request.args.get('offset'))
    ids = request.get_json()['_id']
    user = None
    try:
        user = Database.Users.find_one({"_id":ObjectId(ids)})
    except:
        return {"message": "user id is not correct"}, 401
    keyword = request.args.get('keyword')
    db_response = Database.User.find({
    'username': {
        '$regex': re.compile(rf"{keyword}(?i)")
    }
    })



    for i in db_response:
        i["_id"] = str(i["_id"])
        i["creation_date"] = i["creation_date"].date()
        i["creation_date"] = i["creation_date"].strftime("%Y-%m-%d")
        if 'password' in i:
            del i['password']
        if 'notifications' in i:
            del i['notifications']
        users.append(i)

    count = len(users)

    if (offset > count - 1):
        return jsonify({"users": empty_array}), 204

    

    for i in range(len(users)):
        paginated_list.append(users[i+offset])
        if i+1== limit:
            break

    


 
    for i in user["blockers"]:
        for x in range(len(paginated_list)):
           if i["user_id"] == paginated_list[x]["_id"]:
               del paginated_list[x]
               break
                    


    
    for i in user["blocking"]:
        for x in range(len(paginated_list)):
            if i["user_id"] == paginated_list[x]['_id']:
                del paginated_list[x]
                break

    for i in paginated_list:
        del i["blocking"]
        del i["blockers"]

     


    return jsonify({"users": paginated_list}), 200



    
   