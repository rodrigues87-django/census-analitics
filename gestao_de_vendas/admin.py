from django.contrib import admin
from gestao_de_vendas.models import Produto
from gestao_de_vendas.models import Venda


class ProdutoAdmin(admin.ModelAdmin):
    pass


class VendaAdmin(admin.ModelAdmin):
    pass


admin.site.register(Venda, VendaAdmin)
admin.site.register(Produto, ProdutoAdmin)