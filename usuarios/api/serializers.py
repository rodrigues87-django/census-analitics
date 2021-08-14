from rest_framework.serializers import ModelSerializer
from usuarios.models import User
from django.contrib.auth.models import Group


class UsuariosSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ('id','email', 'password','first_name')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        grupo = Group.objects.get(name='usuarios')
        user.groups.add(grupo)
        user.save()
        return user
