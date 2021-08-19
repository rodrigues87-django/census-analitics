from django.db import models

# Create your models here.
from census.models import Census


class Previsao(Census):
    acertou = models.BooleanField(default=False)

