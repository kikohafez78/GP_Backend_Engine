from flask import request, Response, jsonify, Blueprint
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


update_user = Blueprint('update_user', __name__)



############################################

@update_user.route("/update_profile", methods=["PUT"])
def updateuser():
    # print(request.form["_id"])
    try:
        data = request.get_json()

        name = data["name"]
        date_of_birth = data["date_of_birth"]
        bio = data["bio"]
        location = data["location"]
        website = data["website"]
        prof_pic_url = data["prof_pic_url"]
        cover_pic_url = data["cover_pic_url"]

        user_id = ObjectId(data["_id"])

        myquery1 = {"_id": user_id}

        mydb.User.update_one(
            {"_id": myquery1},
            {"$set": {
                "name": name,
                "date_of_birth": date_of_birth,
                "bio": bio,
                "location": location,
                "website": website,
                "prof_pic_url": prof_pic_url,
                "cover_pic_url": cover_pic_url
            }}

        )

        user = mydb.User.find_one(user_id)
        del user['password']
        user["creation_date"] = user["creation_date"].date()
        user["creation_date"] = user["creation_date"].strftime("%Y-%m-%d")
        user["_id"] = str(user["_id"])
        return Response(
            response=json.dumps(
                {"message": "The request was succesful"
                 }),
            status=200,
            mimetype="application/json")
    except Exception as ex:
        print("**********")
        print(ex)
        print("**********")


