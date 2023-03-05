from django import forms


class DataForm(forms.Form):
    data_from = forms.CharField(label='Дата начала', max_length=10, initial='2019-01-01')
    time_from = forms.CharField(label='Время начала', max_length=8, initial='00:00:00')
    data_to = forms.CharField(label='Дата конца', max_length=10, initial='2019-01-01')
    time_to = forms.CharField(label='Время конца', max_length=8, initial='00:00:00')


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()
