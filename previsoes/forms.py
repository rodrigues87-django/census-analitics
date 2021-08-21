from django.forms import ModelForm
from census.models import Census
from previsoes.models import Previsao


class PrevisaoForm(ModelForm):
    class Meta:
        model = Previsao
        exclude = 'income','acertou'
