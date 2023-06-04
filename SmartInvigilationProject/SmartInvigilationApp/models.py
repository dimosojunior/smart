from django.db import models

# Create your models here.
class InvigilationStaffs(models.Model):

	username = models.CharField(max_length=200,blank=False,null=False)
	camera_no = models.CharField(max_length=200,default="/videos/S1.mp4", blank=False,null=False)
	created = models.DateTimeField(auto_now_add=True)
	updated = models.DateTimeField(auto_now=True)

	def __str__(self):
		return self.username