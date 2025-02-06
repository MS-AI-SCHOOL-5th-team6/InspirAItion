# artwork/views.py
from django.shortcuts import render
from .models import Artwork

def artwork_list(request):
    artworks = Artwork.objects.all()
    return render(request, 'team6/artwork.html', {'artworks': artworks})
#    return render(request, 'artwork/artwork_list.html', {'artworks': artworks})
