from flask import Flask, request, Response, jsonify, Blueprint
import os
import json
import glob2 as gl
import shutil
from Database.Database import Database as mydb
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


Sessions = Blueprint('Sessions', __name__)

allowed_extensions = ["csv", "xlsx", "tsv"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted successfully.")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied: unable to delete file '{file_path}'.")
    except Exception as e:
        print(f"Error occurred while trying to delete file '{file_path}': {e}")

def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and its contents have been deleted successfully.")
    except FileNotFoundError:
        print(f"Directory '{directory_path}' not found.")
    except PermissionError:
        print(f"Permission denied: unable to delete directory '{directory_path}'.")
    except Exception as e:
        print(f"Error occurred while trying to delete directory '{directory_path}': {e}")


@Sessions.route("/", methods = ['POST'])
def create_session():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    user_name = request.form.get("user_name")
    session_name = request.form.get('session_name')
    files = request.files.getlist('files')
    user = mydb.User.find_one({"user_name": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    if mydb.Sessions.find_one({"session_name": session_name.lower()}) is not None:
        return {"message": "session already exists"}, 405
    try:
        os.makedirs(f"Sessions\\{user_name}\\{session_name}")
    except:
        return {"message" : "session already exists"}, 402
    succefull_uploads = {}
    unsuccessfull_uploads = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(os.getcwd(), f"Sessions\\{user_name}\\{session_name}\\{filename}")
            file.save(filepath)
            if len(gl.glob(filepath)) > 0:
                succefull_uploads[filename] = filepath
            else:
                unsuccessfull_uploads.append(filename)
                
    new_session = mydb.Sessions.insert_one({
        "owner": user_name,
        "owner_id": user["_id"],
        "session_name": session_name,
        "messages":[],
        "sheets": succefull_uploads,
        })
    if len(unsuccessfull_uploads) == len(files):
        return Response(response=json.dumps({"message": "The request was unsuccesful, unable to save files, please try agian later",}),status = 500, mimetype="application/json")
    return Response(response=json.dumps({"message": "The request was succefull in creating a new session", "session_id": str(new_session.inserted_id) ,"session_name":session_name,"uploaded_files": list(succefull_uploads.keys()), "unuploaded_files": unsuccessfull_uploads}),status = 200, mimetype="application/json")


@Sessions.route("/workbook", methods = ['PUT'])
def add_new_workbook():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    user_name = request.form.get("user_name")
    session_name = request.form.get('session_name')
    files = request.files.getlist('files')
    user = mydb.User.find_one({"user_name": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    
    succefull_uploads = {}
    unsuccessfull_uploads = []
    for file in files:
        if file and allowed_file(file.filename):
            if len(gl.glob(f"Sessions\\{user_name}\\{session_name}\\{file.filename}")) == 0:
                filename = file.filename
                filepath = os.path.join(os.getcwd(), f"Sessions\\{user_name}\\{session_name}\\{filename}")
                file.save(filepath)
                if len(gl.glob(filepath)) > 0:
                    succefull_uploads[filename] = filepath
                else:
                    unsuccessfull_uploads.append(filename)
            else:
                unsuccessfull_uploads.append(filename)
    
    session.update(succefull_uploads)
    new_session = mydb.Sessions.update_one({
        "owner": user_name,
        "owner_id": user["_id"],
        "session_name": session_name,
        },
        {
         "$set":{
             "sheets": session["sheets"]
             }   
        })
    if len(unsuccessfull_uploads) == len(files):
        return Response(response=json.dumps({"message": "The request was unsuccesful, unable to save files, please try again later",}),status = 500, mimetype="application/json")
    return Response(response=json.dumps({"message": "The request was succefull in adding new files", "session_id": str(session["_id"]) ,"session_name":session_name,"uploaded_files": list(succefull_uploads.keys()), "unuploaded_files": unsuccessfull_uploads}),status = 200, mimetype="application/json")

@Sessions.route("/workbook", methods = ['DELETE'])
def delete_workbooks():
    user_name = request.json['user_name']
    session_name = request.json['session_name']
    file_names = request.json['file_names']
    user = mydb.User.find_one({"user_name": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    file_names = set(file_names)
    names = set([name for name in list(session["sheets"].keys())])
    if not file_names.issubset(names):
        return {"message" : "files do not exist in the current session"}, 400
    
    deleted_files = []
    for file in file_names:
        if session["sheets"].get(file, None) is not None:
            filepath = session["sheets"].get(file)
            del session["sheets"][file]
            delete_file(filepath)
            deleted_files.append(file)
        
    session.update(session["sheets"])
    new_session = mydb.Sessions.update_one({
        "owner": user_name,
        "owner_id": user["_id"],
        "session_name": session_name,
        },
        {
         "$set":{
             "sheets": session["sheets"]
             }   
        })
    if len(deleted_files) == 0:
        return Response(response=json.dumps({"message": "The request was unsuccesful, unable to delete files, please try again later",}),status = 405, mimetype="application/json")
    return Response(response=json.dumps({"message": "The request was succefull in delete files", "session_id": str(session["_id"]) ,"session_name":session_name,"deleted_files": deleted_files}),status = 200, mimetype="application/json")

@Sessions.route("/workbook",methods = ['GET'])
def get_sheet_names():
    user_name = request.form.get('user_name')
    session_name = request.form.get('session_name')
    user = mydb.User.find_one({"user_name": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    names = [name for name in list(session["sheets"].keys())]
    return Response(response=json.dumps({"message": "The request was succefull in retreiving file names", "session_id": str(session["_id"]) ,"session_name":session_name,"files_name": names}),status = 200, mimetype="application/json")


@Sessions.route("/", methods = ['DELETE'])
def delete_session():
    user_name = request.form.get('user_name')
    session_name = request.form.get('session_name')
    user = mydb.User.find_one({"user_name": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    session: dict = mydb.Sessions.find_one({"session_name" : session_name, "owner_id": user["_id"]})
    if  session is None:
        return Response(response=json.dumps({"message": "session doesnt exist"}),status = 400, mimetype="application/json")
    result = mydb.Sessions.delete_one({"session_name" : session_name, "owner_id": user["_id"]})
    delete_directory(f"Sessions\\{user_name}\\{session_name}")
    if result.deleted_count == 0:
        return Response(response=json.dumps({"message": "The request was succefull in retreiving file names", "session_id": str(session["_id"]) ,"session_name":session_name}),status = 400, mimetype="application/json")
    return Response(response=json.dumps({"message": "The request was succefull in deleting session","session_name":session_name}),status = 200, mimetype="application/json")

@Sessions.route("/", methods = ["GET"])
def get_sessions():
    user_name = request.form.get('user_name')
    user = mydb.User.find_one({"user_name": user_name.lower()})
    if user == None:
        return {"message" : "user is not found in the database"}, 404
    sessions: dict = mydb.Sessions.find({ "owner_id": user["_id"]})
    if  len(sessions) == 0:
        return Response(response=json.dumps({"message": "session doesnt exist"}), status = 405, mimetype="application/json")
    return Response(response=json.dumps({"message": "The request was succefull in returing session names", "sessions": sessions}),status = 200, mimetype="application/json")

    
# if __name__ == '__main__':
#     app.run(host='0.0.0.0')
