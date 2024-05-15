from flask_restful import Resource, Api, reqparse
import bcrypt
from flask import Flask, request, Response, jsonify, Blueprint
from bson import json_util
from bson.objectid import ObjectId
import pymongo
import jwt
import datetime
import json
from functools import wraps
from flask_cors import cross_origin
from Database.Database import Database as mydb


change_password = Blueprint('change_password', __name__)




############################################

@change_password.route("/change_password", methods=["PUT"])
def change_pass(current_user):
    # print(request.form["_id"])
    try:
        data = request.get_json()
        password = data["password"]
        password_byte = bytes(password, "ascii")
        hashed_pw = bcrypt.hashpw(password_byte, bcrypt.gensalt())

        user_id = ObjectId(current_user["_id"])

        db_response = mydb.User.update_one(
            {"_id": user_id},
            {"$set": {"password": hashed_pw}}
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


#############################################
# if __name__ == "__main__":
#     app.run(port=9090, debug=True)
