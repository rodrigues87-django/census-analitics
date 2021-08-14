from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth import logout as django_logout

from usuarios.forms import UserForm
from usuarios.models import User

# Create your views here.
from django.views.decorators.csrf import csrf_protect
from rest_framework import generics
from rest_framework.permissions import AllowAny
from .models import User
from usuarios.api.serializers import UsuariosSerializer


class UserCreateAPIView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UsuariosSerializer
    permission_classes = (AllowAny,)


def base(request):
    return render(request, 'base.html')


def login_user(request):
    return render(request, 'login.html')


@csrf_protect
def submit_login(request):
    if request.POST:
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username)
        print(password)
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')
        else:
            messages.error(request, "Usuário e senha inválido. Favor tentar novamente.")
            return redirect('/usuarios/login')


@csrf_protect
def submit_login_google(request):
    print("teste funcao")
    if request.POST and request.is_ajax():
        first_name = request.POST.get('first_name')
        imagem_url = request.POST.get('imagem_url')
        email = request.POST.get('email')

        password = User.objects.make_random_password()

        try:
            user = User.objects.get(email=email)
            user.set_password(password)
            user.save()
            user = authenticate(username=user.email, password=password)
            if user is not None:
                login(request, user)
                print("usuario logado...redirecionado para pagina inicial: " + password)

        except User.DoesNotExist:

            User.objects.create(first_name="first_name", imagem_url="imagem_url", email=email, password=password)
            user = authenticate(username=email, password=password)
            print("Usuario criado e esta autenticado")

    print("não entrou no post")
    return redirect('/')


def administrador(request):
    if request.user.is_authenticated:
        return render(request, 'index.html')
    else:
        return redirect('/pedidos/solicitar')


@login_required
def logout(request):
    if request.user.is_authenticated:
        django_logout(request)
        return redirect('/usuarios/login/')
    else:
        messages.error(request, "Usuário não está logado")
        return redirect('/usuarios/login/')


def list_usuarios(request):
    try:
        usuario = User.objects.get(usuario=request.user.id)
    except User.DoesNotExist:
        usuario = User.objects.create(nome="Minha Dieta", usuario=request.user)

    return render(request, 'usuarios_lista.html', {'usuario': usuario})


def create_usuario(request):
    form = UserForm(request.POST or None)

    if form.is_valid():
        form.save()
        return redirect('list_usuarios')
    return render(request, 'site/usuarios/usuario-form.html', {'form': form})


def update_usuario(request, id):
    usuario = User.objects.get(id=id)
    form = UserForm(request.POST or None, instance=usuario)
    if form.is_valid():
        form.save()
        return redirect('list_usuarios')
    return render(request, 'site/usuarios/usuario-form.html', {'form': form, 'usuario': usuario})


def delete_usuario(request, id):
    usuario = User.objects.get(id=id)

    if request.method == "POST":
        print("delete usuario post")
        usuario.delete()
        return redirect('list_usuarios')

    return render(request, 'site/usuarios/confirm-usuario-delete.html', {'usuario': usuario})
