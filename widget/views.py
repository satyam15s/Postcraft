from django.shortcuts import render

# Create your views here.

def cpx_widget(request):
    return render(request, 'widget/cpx_widget.html')
