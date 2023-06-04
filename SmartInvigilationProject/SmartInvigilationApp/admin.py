from django.contrib import admin
from .models import *
# Register your models here.

class InvigilationStaffsAdmin(admin.ModelAdmin):
	list_display = ['username','camera_no','created','updated']
	list_filter=['created','updated']
	
admin.site.register(InvigilationStaffs,InvigilationStaffsAdmin)