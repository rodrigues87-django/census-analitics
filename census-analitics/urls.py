from django.contrib import admin
from django.urls import path
from census.views import home, census, CensusListView
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
                  path('dashboard/', home),
                  path('census/', CensusListView.as_view()),
                  path('admin/', admin.site.urls),
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) \
              + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
