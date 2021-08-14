01-criar a venv
python -m venv

02-instalar as bibliotecas do requirements.txt

03-preparando migrações
python manage.py makemigrations

04-aplicar as migrações
python manage.py migrate

05-criando superusuario
python manage.py createsuperuser

06-rodando a aplicação
python manage.py runserver



##3-Manipulando objeto

###3.0- Importando classe
```
from documento.models import Documento
```
###3.1 -Criando objeto
```
Documento.objects.create(nome="12351asdasd")
```

###3.2 -Listar
```
documentos = Documento.objects.all()
```

###3.3 -Filtrar
```
documento = Documento.objects.filter(id=1)
```

###3.4 -Atualizar
```
documento.update(nome="novo nome")
```
###3.5 -Deletar
```
documento.delete()
```

## 4- Querysets
### 4.1 - Recuperando o proprio objeto
```
usuario = User.objects.first()
usuario = User.objects.get(first_name="pedro")    =============>      1 valor
usuario = User.objects.filter(first_name="pedro") =============>      lista
```
### 4.2 - Verificando a existencia do objeto
```
try:
    usuario = User.objects.get(email="pedroh.mix@gmail.com")
except User.DoesNotExist:
    usuario = None
```
### 4.3 - Recuperando o atributo do objeto
```
usuario.id
```

### 4.4 - Recuperando a lista de vendas que contém um usuario
```
usuario.venda_set.all()
```
