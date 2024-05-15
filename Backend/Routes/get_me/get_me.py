from flask_restful import Resource, Api, reqparse
from flask import Flask, request, Response, jsonify, Blueprint
from flask_cors import cross_origin
from bson.objectid import ObjectId
import jwt
import json
from functools import wraps
from Database.Database import Database as mydb


get_me = Blueprint('get_me', __name__)



############################################

@get_me.route("/me", methods=["GET"])
def me():
    id = request.get_json()["_id"]
    user = None
    try:
        user = mydb.User.find_one({"_id":ObjectId(id)})
    except:
        return {"message" : "user id is on correct"}, 401
    try:
        user_id = ObjectId(user["_id"])
        user = mydb.User.find_one(user_id)
        if 'password' in user:
            del user['password']
        if 'notifications' in user:
            del user['notifications']
        user["creation_date"] = user["creation_date"].date()
        user["creation_date"] = user["creation_date"].strftime("%Y-%m-%d")
        user["_id"] = str(user["_id"])

        return Response(
            response=json.dumps(
                {"message": "The request was succesful",
                 "user": user
                 }),
            status=200,
            mimetype="application/json")
    except Exception as ex:
        print("**********")
        print(ex)
        print("**********")


#############################################
# if __name__ == "__main__":
#     app.run(port=8081, debug=True)
