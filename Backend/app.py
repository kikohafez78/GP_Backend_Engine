#import imp
from flask import Flask
from flask_cors import CORS


#### User data #####
from Auto_Engine import Auto_Engine
from Auto_Config import get_auto_config
#Karim
from Routes.Login.Login import Login
from Routes.GetAllUsers.GetAll import GetAll
from Routes.Signup.signup import signup, mail
from Routes.get_me.get_me import get_me
#Mohamed 
from Routes.get_user.get_user import get_user
from Routes.update_user.update_user import update_user
from Routes.forgot_password.forgot_password import forgot_password
from Routes.change_password.change_password import change_password
#### Auto Engine ####

# Creating flask application
app = Flask(__name__, template_folder='Templates')
#====== Auto Engine ===========
config = get_auto_config()
Engine = Auto_Engine(config)
#==============================
app.config.from_pyfile('config.cfg')
mail.init_app(app)

app.secret_key = "MakO"





# Registering all the blue prints created in other files

app.register_blueprint(Login, url_prefix='/Login')
app.register_blueprint(GetAll, url_prefix='/users')
app.register_blueprint(signup, url_prefix='/signup')
app.register_blueprint(get_me, url_prefix='/users')
app.register_blueprint(get_user, url_prefix='/users')
app.register_blueprint(update_user, url_prefix='/users')
app.register_blueprint(forgot_password, url_prefix='/users')
app.register_blueprint(change_password, url_prefix='/users')


CORS(app)


# Running the application
if __name__ == '__main__':
    app.run(host='0.0.0.0')
