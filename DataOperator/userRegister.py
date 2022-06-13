import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoProject.settings")
django.setup()

from app0.models import UserModel


from conf.readConfig import readConfig as rc
from DataOperator.jsonOperator import jsonOperator as jo

class mysqlOperator():
    def __init__(self):
        self.dataRootDirPath = rc().getDataRootDirPath()

    def insertUser(self,username,password):
        UserModel(

            username=username,
            password=password,


        ).save()

