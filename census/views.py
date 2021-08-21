from django.http import HttpResponseRedirect
from django.shortcuts import render

# Create your views here.
from django.urls import reverse
from django.views.generic import ListView, CreateView
from rest_framework.renderers import JSONRenderer

from census.forms import CensusForm
from census.models import Census
from machine_learning_model.api.serializers import MachineLearningModelSerializer
from machine_learning_model.constants import ARVORES_DE_DECISAO, REGRESSAO_LOGISTICA
from machine_learning_model.models import MachineLearningModel
from census.services import abrir_census
from previsoes.forms import PrevisaoForm
from previsoes.models import Previsao


class CensusListView(ListView):
    model = Census
    template_name = 'census.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'census_list'  # Default: object_list
    paginate_by = 10
    queryset = Census.objects.all().order_by('id')


class CensusFormView(CreateView):
    model = Previsao
    form_class = PrevisaoForm
    template_name = 'previsoes.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'form'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        model_instance = form.save(commit=False)
        model_instance.save()

        machine_learning_model = MachineLearningModel()

        machine_learning_model.open_classificador(Census, REGRESSAO_LOGISTICA)
        query = Previsao.objects.filter(id=model_instance.id)
        machine_learning_model.get_data(query)
        machine_learning_model.definir_precisao()

        if machine_learning_model.previsao[0] == 1:
            self.previsao = "Salario maior que 50.000$"
        else:
            self.previsao = "Salario menor que 50.000$"

        return render(self.request, 'previsoes.html', {'previsao': self.previsao})


def home(request):
    modelos = MachineLearningModel.objects.all()
    modelos_json = MachineLearningModelSerializer(modelos, many=True)
    content = JSONRenderer().render(modelos_json.data)

    return render(request, 'base.html', {'modelos': modelos, 'modelos_json': content.decode()})


def census(request):
    census_list = Census.objects.all()
    return render(request, 'census.html', {'census_list': census_list})
