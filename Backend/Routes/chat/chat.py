from flask import Flask, request, Response, Blueprint, send_file
from bson.objectid import ObjectId
import json
from Database.Database import Database as mydb
import datetime
import os
import sys
# from app import engine
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def process_message(text, sheet):
    return [], sheet, []

def convert_to_numbered_list(string_array):
    numbered_list = ""
    for index, string in enumerate(string_array, start=1):
        numbered_list += f"{index}. {string}\n"
    return numbered_list.strip()  # Remove the trailing newline


chat = Blueprint('chat', __name__)


@chat.route("/", methods=["POST"])
def send_message():
    user_name = request.form.get('user_name')
    session_name = request.form.get('session_name')
    user = mydb.User.find_one({"username": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    sheet_name = request.form.get('sheet_name')
    text = request.form.get('prompt')
    if sheet_name.lower() not in list(session["sheets"].keys()):
        return Response(response=json.dumps({"message": "The request was succefull in retreiving file names", "session_id": str(session["_id"]) ,"session_name":session_name}),status = 200, mimetype="application/json")
    
    #================================================
    file = open(session["sheets"][sheet_name],"r")
    steps, updated_sheet, errors = process_message(text, file)
    updated_sheet.save(session["sheets"][session["sheets"][sheet_name]])
    #================================================
    user_message = {
        "id" : len(session["messages"]) + 1,
        "role": "user",
        "text": text,
        "target_sheet": sheet_name,
        "date": str(datetime.datetime.now())
    }
    system_message = {
        "id" : len(session["messages"]) + 2,
        "role": "Auto",
        "text": convert_to_numbered_list(steps),
        "target_sheet": sheet_name,
        "date": str(datetime.datetime.now())
    }
    session["messages"].append([user_message, system_message])
    mydb.Sessions.update_one({"session_name": session_name, "owner_id": user["_id"]}, {"$set": session})
    #================================================
    if len(steps) == 0:
        return Response(response=json.dumps({"message": "The request was could'nt be fullfilled due to the errors shown", "session_id": str(session["_id"]) ,"session_name":session_name,"errors": errors}),status = 405, mimetype="application/json")
    resp = send_file(
        os.path.join(os.getcwd(), f"Sessions\\{user_name}\\{session_name}\\{sheet_name}"),
        as_attachment=True,
        attachment_filename=sheet_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    resp.headers['message'] = 'The request was succefull in fullfilling the tasks required'
    resp.headers['step'] = json.dumps(steps)
    resp.status = 200
    return resp
        
@chat.route("/",methods = ["GET"])
def get_chat():
    user_name = request.form.get('user_name')
    session_name = request.form.get('session_name')
    user = mydb.User.find_one({"username": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    return Response(response=json.dumps({"message":"the request is successfully fullfilled", "chat": session["messages"]}), status=200, mimetype = "application/json")

@chat.route("/", methods = ['PUT'])
def edit_message():
    user_name = request.form.get('user_name')
    session_name = request.form.get('session_name')
    message_id = request.form.get('message_id')
    text = request.form.get('prompt')
    user = mydb.User.find_one({"username": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    user_message = None
    system_message = None
    i = 0
    messages: list = session["messages"]
    for msg in messages:
        if msg["id"] == message_id:
            user_message = msg
            system_message = messages[i + 1]
            i = int(msg["id"])
            break
        i += 1
    if user_message is None:
        return Response(response=json.dumps({"message": "message doesnt exist"}),status = 405, mimetype="application/json")
    if request.form.get("sheet_name", None) is not None:
        if request.form.get("sheet_name", None).lower() not in list(session["sheets"].keys()):
            return Response(response=json.dumps({"message": "The request was succefull in retreiving file names", "session_id": str(session["_id"]) ,"session_name":session_name}),status = 200, mimetype="application/json")
        user_message["target_sheet"] = request.form.get("sheet_name", None)
        system_message["target_sheet"] = request.form.get("sheet_name", None)
    #======================================
    file = open(session["sheets"][user_message["target_sheet"]],"r")
    steps, updated_sheet, errors = process_message(text, file)
    updated_sheet.save(session["sheets"][session["sheets"][user_message["target_sheet"]]])
    #======================================
    user_message["text"] = text
    system_message["text"] = convert_to_numbered_list(steps)
    session["messages"].append([user_message, system_message])
    session["messages"][i] = user_message
    session["messages"][i + 1] = system_message
    mydb.Sessions.update_one({"session_name": session_name, "owner_id": user["_id"]}, {"$set": session})
    #================================================
    if len(steps) == 0:
        return Response(response=json.dumps({"message": "The request was could'nt be fullfilled due to the errors shown", "session_id": str(session["_id"]) ,"session_name":session_name,"errors": errors}),status = 405, mimetype="application/json")
    resp = send_file(
        os.path.join(os.getcwd(), f"Sessions\\{user_name}\\{session_name}\\{user_message['target_sheet']}"),
        as_attachment=True,
        attachment_filename=user_message["target_sheet"],
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    resp.headers['message'] = 'The request was succefull in editing the message'
    resp.headers['step'] = json.dumps(steps)
    resp.status = 200
    return resp


@chat.route("/", methods = ['DELETE'])
def delete_message():
    user_name = request.form.get('user_name')
    session_name = request.form.get('session_name')
    message_id = request.form.get('message_id')
    user = mydb.User.find_one({"username": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    user_message = None
    system_message = None
    i = 0
    messages: list = session["messages"]
    for msg in messages:
        if msg["id"] == message_id:
            user_message = msg
            system_message = messages[i + 1]
            i = int(msg["id"])
            break
        i += 1
    if user_message is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}), status = 405, mimetype="application/json")
    session["messages"].remove(user_message)
    session["messages"].remove(system_message)
    i -= 2
    for index in range(i, len(messages)):
        session["messages"]["id"] = index    
    mydb.Sessions.update_one({"session_name": session_name, "owner_id": user["_id"]}, {"$set": session})
    return Response(response=json.dumps({"message": "messages have been deleted", "new_chat": session["messages"]}),status = 200, mimetype="application/json")



