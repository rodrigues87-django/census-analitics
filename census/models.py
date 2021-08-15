from django.db.models import *


class Census(Model):
    age = IntegerField()
    workclass = CharField(max_length=20)
    final_weight = IntegerField()
    education = CharField(max_length=20)
    education_num = IntegerField()
    marital_status = CharField(max_length=20)
    occupation = CharField(max_length=20)
    relationship = CharField(max_length=20)
    race = CharField(max_length=20)
    sex = CharField(max_length=20)
    capital_gain = IntegerField()
    capital_loos = IntegerField()
    hour_per_week = IntegerField()
    native_country = CharField(max_length=20)
    income = CharField(max_length=20)

    def __int__(self):
        return self.id

    class Meta:
        verbose_name_plural = "Census"

