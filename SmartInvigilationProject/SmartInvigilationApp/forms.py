
from .models import *
from django.forms import ModelForm
from django import forms


class InvigilationStaffsForm(forms.ModelForm):

	class Meta:
		model = InvigilationStaffs
		fields ='__all__'