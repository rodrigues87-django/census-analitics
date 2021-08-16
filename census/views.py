from django.forms import ModelForm
from django.shortcuts import render
import json

# Create your views here.
from django.core import serializers
from django.views.generic import ListView, TemplateView, FormView, CreateView
from rest_framework.renderers import JSONRenderer

from census.forms import CensusForm
from census.models import Census
from machine_learning_model.api.serializers import MachineLearningModelSerializer
from machine_learning_model.models import MachineLearningModel


class CensusListView(ListView):
    model = Census
    template_name = 'census.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'census_list'  # Default: object_list
    paginate_by = 10
    queryset = Census.objects.all().order_by('id')


class CensusFormView(CreateView):
    model = Census
    form_class = CensusForm
    template_name = 'previsoes.html'  # Default: <app_label>/<model_name>_list.html
    context_object_name = 'form'

    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        return super().form_valid(form)


def home(request):
    modelos = MachineLearningModel.objects.all()
    modelos_json = MachineLearningModelSerializer(modelos, many=True)
    content = JSONRenderer().render(modelos_json.data)

    return render(request, 'base.html', {'modelos': modelos, 'modelos_json': content.decode()})


def census(request):
    census_list = Census.objects.all()
    return render(request, 'census.html', {'census_list': census_list})
