#import imp
from flask import Flask
from flask_cors import CORS

import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
#### User data #####
# from Auto_Engine import Auto
# from Auto_Config import get_auto_config
#Karim
from Routes.Login.Login import Login
from Routes.GetAllUsers.GetAll import GetAll
from Routes.Signup.signup import signup
from Routes.get_me.get_me import get_me
#Mohamed 
from Routes.get_user.get_user import get_user
from Routes.update_user.update_user import update_user
from Routes.forgot_password.forgot_password import forgot_password
from Routes.change_password.change_password import change_password
#sessions
from Routes.Session.workbooks import Sessions
#chat
from Routes.chat.chat import chat
#### Auto Engine ####
from Auto_Engine import Auto
from Auto_Config import get_auto_config
# Creating flask application
app = Flask(__name__, template_folder='Templates')
#====== Auto Engine ===========
config = get_auto_config()
app.config['engine'] = Auto(config)
#==============================
app.config.from_pyfile('config.cfg')


# Registering all the blue prints created in other files

app.register_blueprint(Login, url_prefix='/login')
app.register_blueprint(GetAll, url_prefix='/users')
app.register_blueprint(signup, url_prefix='/signup')
app.register_blueprint(get_me, url_prefix='/users')
app.register_blueprint(get_user, url_prefix='/users')
app.register_blueprint(update_user, url_prefix='/users')
app.register_blueprint(forgot_password, url_prefix='/users')
app.register_blueprint(change_password, url_prefix='/users')
app.register_blueprint(Sessions, url_prefix="/sessions")
app.register_blueprint(chat, url_prefix="/chat")
# CORS(app)


# Running the application
if __name__ == '__main__':
    app.run(host='0.0.0.0')
