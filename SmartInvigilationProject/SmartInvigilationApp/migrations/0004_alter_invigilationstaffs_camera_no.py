# Generated by Django 4.1.3 on 2023-06-03 06:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SmartInvigilationApp', '0003_alter_invigilationstaffs_camera_no'),
    ]

    operations = [
        migrations.AlterField(
            model_name='invigilationstaffs',
            name='camera_no',
            field=models.CharField(default='/videos/S1.mp4', max_length=200),
        ),
    ]