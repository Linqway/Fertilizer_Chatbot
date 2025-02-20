import os
import sys

from pathlib import Path

sys.path.append(str(Path(os.path.realpath(os.path.dirname(__file__))).parent)) 
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

import django
from django.conf import settings
settings.INSTALLED_APPS.remove('api.apps.ApiConfig')
settings.INSTALLED_APPS.remove('FertilizerChatbot')

django.setup()

from django.contrib.auth.models import User, Group

def prRed(skk): print("\033[91m{}\033[00m" .format(skk),end='') 
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk),end='') 
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk),end='')
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk),end='') 

def create_user():

    import django.contrib.auth
    User = django.contrib.auth.get_user_model()

    username =input("Enter Username : ")

    if User.objects.filter(username=username).exists():
        prRed("\n\nUsername already exists! please use another username\n\n")
        return

    password =input("Enter Password : ")
    repassword =input("Retype Password : ")

    if(password == repassword):
        user = User.objects.create_user(username, password=password)
        user.is_superuser = False
        user.is_staff = False
        user.save()
        print("\n\nCreated User : "+username+"\n\n")
    else:
        prRed("\n\nPasswords do not match! Try again\n\n")

def delete_user():
    username =input("Enter Username : ")
    try:
        user = User.objects.get(username=username)
        choice = input("Are you sure want to delte [Yy/Nn] : ")
        if(choice in ['Y','y']):
            user.delete()
            prGreen("\n\nusername "+username+" Successfully deleted!\n\n")
        else:
            prYellow("\n\nOperation aborted\n\n")
    except User.DoesNotExist:
        prRed("\n\nUsername Does not exists. Type Correct username\n")


while True:  
    print("\n--MAIN MENU--\n")  
    print("1. Create User")  
    print("2. Delete User")  
    print("3. Exit")
    choice = int(input("Enter the Choice : "))

    if choice == 1:
        create_user()
    elif choice == 2:
        delete_user()
    else :
        break
