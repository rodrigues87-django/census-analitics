from django.db import models
from usuarios.models import User


class Produto(models.Model):
    descricao = models.CharField(max_length=30)
    preco = models.DecimalField(max_digits=5, decimal_places=2)

    def __str__(self):
        return self.descricao


class Venda(models.Model):
    valor = models.DecimalField(max_digits=5, decimal_places=2)
    pessoa = models.ForeignKey(User, on_delete=models.CASCADE)
    produtos = models.ManyToManyField(Produto, blank=True)

    def __int__(self):
        return self.id
