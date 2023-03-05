# Импорт
from django.shortcuts import render

from .forms import DataForm

from tec_func.calculation import NOX, graphic, maxValue, download_data, input_data
from datetime import datetime
import io
import csv


# Переменные для хранения данных
data = {}
file_name = 'нет файла'


def home_page(request):
    # Вывод на страницу инфомрации
    context = {
        'data': "Дата: " + datetime.today().strftime('%Y-%m-%d'),
    }

    # Отрисовка страницы
    return render(request=request, template_name='home.html', context=context)


def tube_1_page(request):
    global file_name, data

    # Форма для загрузки
    if request.method == 'POST' and 'upload_phrase' in request.POST:
        file = request.FILES['file']
        file_name = file.name
        read_data = file.read().decode('utf-8')
        raw_data = list(csv.reader(io.StringIO(read_data)))
        data = input_data(raw_data)

    # Форма для данных
    data_form = DataForm(prefix='data')

    # Форма для выгрузки
    if request.method == 'POST' and 'download_phrase' in request.POST:
        download_data(data)

    # Вывод на страницу инфомрации
    context = {
        'data': "Дата: " + datetime.today().strftime('%Y-%m-%d'),
        'file_name': file_name,
        'data_form': data_form,
    }

    # Отрисовка страницы
    return render(request=request, template_name='tube_1.html', context=context)


def tube_2_page(request):
    global file_name, data

    # Форма для загрузки
    if request.method == 'POST' and 'upload_phrase' in request.POST:
        file = request.FILES['file']
        file_name = file.name
        read_data = file.read().decode('utf-8')
        raw_data = list(csv.reader(io.StringIO(read_data)))
        data = input_data(raw_data)

    # Форма для данных
    data_form = DataForm(prefix='data')

    # Форма для выгрузки
    if request.method == 'POST' and 'download_phrase' in request.POST:
        download_data(data)

    # Вывод на страницу инфомрации
    context = {
        'data': "Дата: " + datetime.today().strftime('%Y-%m-%d'),
        'file_name': file_name,
        'data_form': data_form,
    }

    # Отрисовка страницы
    return render(request=request, template_name='tube_2.html', context=context)


def tube_3_page(request):
    global file_name, data

    # Форма для загрузки
    if request.method == 'POST' and 'upload_phrase' in request.POST:
        file = request.FILES['file']
        file_name = file.name
        read_data = file.read().decode('utf-8')
        raw_data = list(csv.reader(io.StringIO(read_data)))
        data = input_data(raw_data)

    # Форма для данных
    data_form = DataForm(prefix='data')

    # Форма для выгрузки
    if request.method == 'POST' and 'download_phrase' in request.POST:
        download_data(data)

    # Вывод на страницу инфомрации
    context = {
        'data': "Дата: " + datetime.today().strftime('%Y-%m-%d'),
        'file_name': file_name,
        'data_form': data_form,
    }

    # Отрисовка страницы
    return render(request=request, template_name='tube_3.html', context=context)


def graphic_page(request):
    global data, graph, date_from, time_from, date_to, time_to
    # Форма для даты
    if request.method == 'POST' and 'data_phrase' in request.POST:
        data_form = DataForm(request.POST, prefix='data')
        if data_form.is_valid():
            date_from = data_form.cleaned_data['data_from']
            time_from = data_form.cleaned_data['time_from']
            date_to = data_form.cleaned_data['data_to']
            time_to = data_form.cleaned_data['time_to']
            graph = graphic(date_from + ' ' + time_from,
                            date_to + ' ' + time_to, data)

    # Вывод на страницу информации
    context = {
        'data': "Дата: " + datetime.today().strftime('%Y-%m-%d'),
        'graphic': graph,
        'data1': date_from + ' ' + time_from,
        'data2': date_to + ' ' + time_to,
        'NOxmax': '{0:.2f}'.format(maxValue(data, date_from + ' ' + time_from, date_to + ' ' + time_to)[0]),
        'datamax': maxValue(data, date_from + ' ' + time_from, date_to + ' ' + time_to)[1]
    }

    # Отричсовка страницы
    return render(request=request, template_name='graphic.html', context=context)
