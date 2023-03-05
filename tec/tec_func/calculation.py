import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mpld3
import pandas as pd
from mpld3.utils import get_id
import collections

from datetime import datetime

import numpy

from plotly.offline import plot
from plotly.graph_objects import Scatter, Figure

import configparser


def input_data(raw_data):
    data = {}
    flag = True
    for row in raw_data:
        if flag:
            for key in row:
                data[key] = []
            flag = False
            continue

        for d, key in zip(row, data.keys()):
            data[key].append(d)

    # Перевод даты в файле в нормальный вид
    # 2019-01-01 00:00:00
    for i in range(len(data['Time'])):
        data['Time'][i] = data['Time'][i][:4] + '-' + data['Time'][i][8:10] + '-' + data['Time'][i][5:7] + data['Time'][
                                                                                                               i][10:19]

    return data


config = configparser.ConfigParser()
config.read("config.ini")

B_sg = 0.95  ## зависимость степени выгорания топлива (таблица)
# B_p = 10.79  ## расчетный расход топлива
Q_i_r = 35.3  ## Теплота сгорания
g_f = 0  ## удельный расход пара через форсунку на 1 кг мазута
i_f = 0  ## энтальпия пара, подаваемого на распыл
a2_t = 1.05  ## (2 - степень производной) коэффициент избытка воздуха на выходе из топки
delta_a_t = 0  ## присосы холодного воздуха в топку
I_v_o = 4.631  ## энтальпия теоретически необходимого количества воздуха при температуре горячего воздуха
I_xv_o = 0.378  ## энтальпия теоретически необходимого количества воздуха при температуре холодного воздуха
K_R = 0.85  ## способ ввода газов рециркуляции (таблица)
a_otb = 1  ## коэффициент избытка воздуха в месте отбора газов из конвективного газохода на рециркуляцию
R = 0.2  ## доля рециркуляции дымовых газов в зону активного горения
r = 0  ## теплота парообразования
i_vl = 0  ## энтальпия влаги, поступающей в зону активного горения
a_t = 2.7  ## ширина топки
b_t = 7.534  ## глубина топки
V_g_o = 29.88  ## объем продуктов сгорания, образовавшихся при стехиометрическом сжигании топлива
V_v_o = 9.73  ## теоретический объем воздуха и продуктов сгорания
p_gaz_o = 0.712  ## из формулы g-водотопливное отношение в долях (g = G_vl / G_gaz * p_gaz_o)
psi1 = 0.1  ## известная константа
psi = [1, 2, 3]  ## тепловая эффективность участка
F_st = [1, 2, 3]  ## площадь участка стены ЗАГ
F_st2 = 17  ## из рисунков топок
F_up = 18  ## из рисунков топок
F_down = 19  ## из рисунков топок
F_f_n = 27.89  ## площадь фронтового экрана ниже ЗАГ
F_b_n = 13.85  ## площадь боковых экрана ниже ЗАГ
F_z_n = 27.89  ## площадь заднего экрана ниже ЗАГ
F_p = 211.97  ## площадь пода
psi_f = 0.6  ## тепловая эффективность фронтового экрана ниже ЗАГ
psi_b = 0.7  ## тепловая эффективность боковых экрана ниже ЗАГ
psi_z = 0.8  ## тепловая эффективность заднего экрана ниже ЗАГ
psi_p = 0.9  ## тепловая эффективность пода
I_gr_o = 6.278  ## энтальпия газов рециркуляции
some_eps = 0.75  ## коэффициент заполнения топочной камеры восходящими потоками газов
b_sg = 0.95  ## зависимость степени выгорания топлива от коэффициента избытка воздуха в ЗАГ
a_t_streak = 1.05  ## коэффициент избытка воздуха на выходе из топки
a_t_delta = 0  ## присосы холодного воздуха в топку
g = 0.121  ## g-водотопливное отношение в долях
K_g = 0.75  ## коэффициент зависит от конструкции горелки
K_p = 21.1  ## поправочный коэффициент

"""B_sg = config['Tube']['B_sg']  ## зависимость степени выгорания топлива (таблица)
# B_p = 10.79  ## расчетный расход топлива
Q_i_r = config['Tube']['Q_i_r']  ## Теплота сгорания
g_f = config['Tube']['g_f']  ## удельный расход пара через форсунку на 1 кг мазута
i_f = config['Tube']['i_f']  ## энтальпия пара, подаваемого на распыл
a2_t = config['Tube']['a2_t']  ## (2 - степень производной) коэффициент избытка воздуха на выходе из топки
delta_a_t = config['Tube']['delta_a_t']  ## присосы холодного воздуха в топку
I_v_o = config['Tube']['I_v_o']  ## энтальпия теоретически необходимого количества воздуха при температуре горячего воздуха
I_xv_o = config['Tube']['I_xv_o']  ## энтальпия теоретически необходимого количества воздуха при температуре холодного воздуха
K_R = config['Tube']['K_R']  ## способ ввода газов рециркуляции (таблица)
a_otb = config['Tube']['a_otb']  ## коэффициент избытка воздуха в месте отбора газов из конвективного газохода на рециркуляцию
R = config['Tube']['R']  ## доля рециркуляции дымовых газов в зону активного горения
r = config['Tube']['r']  ## теплота парообразования
i_vl = config['Tube']['i_vl']  ## энтальпия влаги, поступающей в зону активного горения
a_t = config['Tube']['a_t']  ## ширина топки
b_t = config['Tube']['b_t']  ## глубина топки
V_g_o = config['Tube']['V_g_o']  ## объем продуктов сгорания, образовавшихся при стехиометрическом сжигании топлива
V_v_o = config['Tube']['V_v_o']  ## теоретический объем воздуха и продуктов сгорания
p_gaz_o = config['Tube']['p_gaz_o']  ## из формулы g-водотопливное отношение в долях (g = G_vl / G_gaz * p_gaz_o)
psi1 = config['Tube']['psi1']  ## известная константа
psi = config['Tube']['psi']  ## тепловая эффективность участка
F_st = config['Tube']['F_st']  ## площадь участка стены ЗАГ
F_st2 = config['Tube']['F_st2']  ## из рисунков топок
F_up = config['Tube']['F_up']  ## из рисунков топок
F_down = config['Tube']['F_down']  ## из рисунков топок
F_f_n = config['Tube']['F_f_n']  ## площадь фронтового экрана ниже ЗАГ
F_b_n = config['Tube']['F_b_n']  ## площадь боковых экрана ниже ЗАГ
F_z_n = config['Tube']['F_z_n']  ## площадь заднего экрана ниже ЗАГ
F_p = config['Tube']['F_p']  ## площадь пода
psi_f = config['Tube']['psi_f']  ## тепловая эффективность фронтового экрана ниже ЗАГ
psi_b = config['Tube']['psi_b']  ## тепловая эффективность боковых экрана ниже ЗАГ
psi_z = config['Tube']['psi_z']  ## тепловая эффективность заднего экрана ниже ЗАГ
psi_p = config['Tube']['psi_p']  ## тепловая эффективность пода
I_gr_o = config['Tube']['I_gr_o']  ## энтальпия газов рециркуляции
some_eps = config['Tube']['some_eps']  ## коэффициент заполнения топочной камеры восходящими потоками газов
b_sg = config['Tube']['b_sg']  ## зависимость степени выгорания топлива от коэффициента избытка воздуха в ЗАГ
a_t_streak = config['Tube']['a_t_streak']  ## коэффициент избытка воздуха на выходе из топки
a_t_delta = config['Tube']['a_t_delta']  ## присосы холодного воздуха в топку
g = config['Tube']['g']  ## g-водотопливное отношение в долях
K_g = config['Tube']['K_g']  ## коэффициент зависит от конструкции горелки
K_p = config['Tube']['K_p']  ## поправочный коэффициент"""


# коэффициент корректировки подачи объема воздуха по отношению к объему подаваемого топлива KK_POD_VOZD= FuelGasGTYConsumption/AirFurnaceConsumption
def getKoeff(B_p, AFC):
    return B_p / AFC


def NOX(B_p, AFC):  ## главная формула (4.1 стр. 14)  ## коэффициент K_g зависит от конструкции горелки
    return 1000 * K_p * 2.05 * 10 ** (-3) * K_g * (26 * numpy.exp(0.26 * (T_zag_streak() - 1700) / 100) - 4.7) * (
            numpy.exp(q_zag_otr(B_p)) - 1) * (13.0 - 79.8 * (a_zag() - 1.07) ** 4 + 18.1 *
                                              (a_zag() - 1.07) ** 3 + 59.4 * (a_zag() - 1.07) ** 2 + 9.6 * (
                                                      a_zag() - 1.07)) * tau_zag(B_p)


def T_zag_streak():  ## 4.4 стр. 15
    return T_ad() * (1 - psi_zag(3)) ** 0.25


def Q_f():  ## 4.8 стр. 16
    return g_f * i_f


def I_gr():  ## 4.11 стр. 17
    return I_gr_o + (a_otb - 1) * I_v_o


def a_g():  ## 4.9 (описание) стр. 17
    return a_t_streak - a_t_delta


def v_ad():  # коэф. заменен
    return -1837
    # return 1059


## 12/11 Принимаем v_ad() равным 1059
def k_t():  ## 4.15-4.16 (описание) стр. 18
    return (v_ad() - 1200) / 1000


def c_g():  ## 4.15 стр. 18
    return (1.57 + 0.134 * k_t()) * 10 ** (-3)


def c_v():  ## 4.17 стр. 18
    return (1.46 + 0.092 * k_t()) * 10 ** (-3)


def c_vl():  ## 4.18 стр. 18
    return 4.1868 * (0.356 - 0.769 * 10 ** (-11) * (v_ad() ** 3) +
                     0.245 * 10 ** (-7) * v_ad() ** 2 + 0.386 * 10 ** (-4) * v_ad()) * 10 ** (-3)


def T_ad():
    return (b_sg * Q_i_r + Q_f() + a_g() * I_v_o + a_t_delta * I_xv_o + K_R * R * I_gr_o + g * (i_vl - r)) / (
            b_sg * V_g_o * c_g() + 1.0161 * (a_zag() - b_sg) * V_v_o * c_v() + 1.24 * g * c_vl() + K_R * R * (
            V_g_o * c_g() + 1.0161 * (a_otb - 1) * V_v_o * c_v() + 1.24 * g * c_vl())) + 273
    # return 1250


def q_zag_otr(B_p):  ## 4.22 стр. 19
    return q_zag(B_p) * (1 - psi_zag(3))


def q_zag(B_p):  ## 4.23 стр.20
    return (B_p * (B_sg * Q_i_r + Q_f() + Q_v() + Q_gr() + Q_vl())) / f_zag()


def psi_zag(n):  ## 4.20 стр. 18 (запросили)
    # sum = 0
    # for i in range(n):
    #    sum = sum + psi[i] * F_st[i]
    # return (sum + psi1 * F_up + psi2() * F_down) / (F_st2 + F_up + F_down)
    return 0.432


def f_zag():  ## 4.24 стр. 20
    return 2 * a_t * b_t + 2 * (a_t + b_t) * h_zag()


def Q_f():  ## g_f при сжигании мазута  4.8 стр. 16
    return g_f * i_f


def Q_v():  ## 4.9 стр. 16
    return a_g() * I_v_o + 0.5 * delta_a_t * I_xv_o


def a_g():  ## 4.9 (см. описание) стр. 17
    return a2_t - delta_a_t


def Q_gr():  ## 4.10 стр. 17
    return K_R * R * I_gr()


def I_gr():  ## 4.11 стр. 17
    return I_gr_o + I_v_o * (a_otb - 1)


def Q_vl():  ## 4.12 стр. 17
    return g * (i_vl - r)


def h_zag():  ## 4.25 стр. 20
    return h_zag_o() * V_g_Rg() / V_g()


def h_zag_o():  ## (коэфф. запросили) 4.26(а,б) стр. 20
    return 11.33


def V_g_Rg():  ## 4.28 стр. 21
    return B_sg * V_g_o + 1.0161 * (a_zag() - B_sg) * V_v_o + 1.24 * g + K_R * R * (
            V_g_o + 1.0161 * (a_otb - 1) * V_v_o + 1.24 * g)


def V_g():  ## 4.27 стр. 21
    return B_sg * V_g_o + 1.0161 * (a_zag() - B_sg) * V_v_o


def a_zag():  ## 4.14 стр. 17
    return a_g() + 0.5 * delta_a_t


def psi2():  ## 4.20 стр. 18
    # return (F_f_n * psi_f + 2 * F_b_n * psi_b + F_z_n * psi_z + F_p * psi_p) / (F_f_n + 2 * F_b_n + F_z_n + F_p)
    return 0.255


def tau_zag(B_p):  ## 4.29 стр. 21
    if B_p == 0:
        B_p = 10.79
    return (a_t * b_t * h_zag()) * some_eps / (B_p * V_g_Rg() * (T_zag_streak() / 273))


def binary_search(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a['Time'])
    while lo < hi:
        mid = (lo + hi) // 2
        midval = a['Time'][int(mid)]
        if midval < x:
            lo = mid + 1
        elif midval > x:
            hi = mid
        else:
            return mid
    return -1


def graphic(startTime, endTime, data):
    data1 = {}
    data1['NOX'] = []

    index_start = binary_search(data, startTime)

    for key1, key2 in zip(data['FuelGasGTYConsumption'][index_start:], data['AirFurnaceConsumption'][index_start:]):
        # for key1, key2 in zip(data['FuelGasGTYConsumption'], data['AirFurnaceConsumption']):
        if data['Time'][data['FuelGasGTYConsumption'].index(key1)] > endTime:
            break
        data1['NOX'].append(NOX(float(key1), float(key2)))

    data2 = []
    for index, item in enumerate(data['СoncentrationNOx'], index_start):
        if data['Time'][data['СoncentrationNOx'].index(item)] > endTime:
            break
        data2.append(float(item) * 1.53)

    # Рисование при помощи matplotlib
    """fig, ax = plt.subplots()
    plt.figure(figsize=(100, 50))
    ax.grid(True, alpha = 1)
    l1 = ax.plot(data['Time'][data['Time'].index(startTime):data['Time'].index(endTime)], data2[data['Time'].index(startTime):data['Time'].index(endTime)], label='Предоставленные данные')

    ax.set_xlim(startTime, endTime)
    ax.set_xticks([])

    l2 = ax.plot(data['Time'][data['Time'].index(startTime):data['Time'].index(endTime)], data1['NOX'][data['Time'].index(startTime):data['Time'].index(endTime)], label='Расчет по методике')

    fig.set_size_inches(14,5)
    ax.set_xlabel('Время')
    ax.set_ylabel('NOX')"""

    # Сохранение в картинку и в html код
    """plt.savefig('tec_web/static/images/graphic.jpeg') # сохраняет картинку
    graph = mpld3.fig_to_html(fig, template_type='general') # преобразует график в html страницу (меняйте fig - это не трогать)"""

    # Рисование при помощи plotly
    fig = Figure()

    g1 = Scatter(x=data['Time'][data['Time'].index(startTime):data['Time'].index(endTime) + 1],
                 y=data2[data['Time'].index(startTime):data['Time'].index(endTime) + 1],
                 mode='lines', name='Фактические данные')
    g2 = Scatter(x=data['Time'][data['Time'].index(startTime):data['Time'].index(endTime) + 1],
                 # y=data1['NOX'][data['Time'].index(startTime):data['Time'].index(endTime) + 1],
                 y=data1['NOX'],
                 mode='lines', name='Расчет по методике')

    fig.add_trace(g1)
    fig.add_trace(g2)
    fig.update_layout(
        autosize=False,
        width=1500,
        height=700,
        paper_bgcolor="White",
        plot_bgcolor='White',
        xaxis=dict(
            title='Время',
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            gridwidth=1,
            gridcolor='Gray',
        ),
        yaxis=dict(
            title='NOx',
            showgrid=True,
            zeroline=True,
            showline=True,
            gridwidth=1,
            gridcolor='Gray',
        )
    )
    graph = plot(fig, output_type='div')

    return graph


def maxValue(data, startTime, endTime):
    max_val = '0'
    date = startTime

    #index_start = binary_search(data, startTime)

    #for key, value in enumerate(data['Time'][index_start:]):
    for key, value in enumerate(data['Time']):
        if value > endTime:
            break
        if data['СoncentrationNOx'][key] > max_val and value >= startTime:
            max_val = data['СoncentrationNOx'][key]
            date = value

    return [float(max_val) * 1.53, date]


def download_data(data):
    data1 = {}
    data1['NOX'] = []

    dataq1 = []
    dataq2 = []
    dataq3 = []
    dataq4 = []
    dataq5 = []
    dataq6 = []
    dataq7 = []
    dataq8 = []
    for key1, key2 in zip(data['FuelGasGTYConsumption'], data['AirFurnaceConsumption']):
        data1['NOX'].append(float(NOX(float(key1), float(key2))))
        dataq1.append(q_zag_otr(float(key1)))
        dataq2.append(q_zag(float(key1)))
        dataq3.append(tau_zag(float(key1)))
        try:
            dataq4.append((float(NOX(float(key1), float(key2))) - 1.53 * float(data['СoncentrationNOx'][data['FuelGasGTYConsumption'].index(key1)])) / (1.53 * float(data['СoncentrationNOx'][data['FuelGasGTYConsumption'].index(key1)])))
        except ZeroDivisionError:
            dataq4.append(0)
        dataq5.append(1.53 * float(data['СoncentrationNOx'][data['FuelGasGTYConsumption'].index(key1)]))
        dataq6.append(float(data['FuelGasGTYConsumption'][data['FuelGasGTYConsumption'].index(key1)]))
        dataq7.append(data['Time'][data['FuelGasGTYConsumption'].index(key1)][:10])
        dataq8.append(data['Time'][data['FuelGasGTYConsumption'].index(key1)][11:])
        #dataq4.append( float(NOX(float(key1), float(key2))) - 1.53 * float(data['СoncentrationNOx'][data['FuelGasGTYConsumption'].index(key1)]) )

    # print(data['Time'][:10])
    # print(','.join(data['Time'][:10]))
    df1 = pd.DataFrame({##'Time': data['Time'],
                        'Time': dataq8,
                        'Data': dataq7,
                        'NOX': data1['NOX'],
                        'СoncentrationNOx': dataq5,
                        'FuelGasGTYConsumption': dataq6,
                        'q_zag_otr': dataq1,
                        'q_zag': dataq2,
                        'tau_zag': dataq3,
                        'accuracy': dataq4})
    
    df = pd.DataFrame({'T_zag_streak': [T_zag_streak()],
                       'Q_f': [Q_f()],
                       'I_gr': [I_gr()],
                       'a_g': [a_g()],
                       'v_ad': [v_ad()],
                       'k_t': [k_t()],
                       'c_g': [c_g()],
                       'c_v': [c_v()],
                       'c_vl': [c_vl()],
                       'T_ad': [T_ad()],
                       'psi_zag': [psi_zag(1)],
                       'f_zag': [f_zag()],
                       'Q_f': [Q_f()],
                       'Q_v': [Q_v()],
                       'Q_gr': [Q_gr()],
                       'I_gr': [I_gr()],
                       'Q_vl': [Q_vl()],
                       'h_zag': [h_zag()],
                       'h_zag_o': [h_zag_o()],
                       'V_g_Rg': [V_g_Rg()],
                       'V_g': [V_g()],
                       'a_zag': [a_zag()],
                       'psi2': [psi2()],})

    filepath = Path('tec_web/downloads/' + datetime.today().strftime('%Y-%m-%d') + '_const.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

    filepath = Path('tec_web/downloads/' + datetime.today().strftime('%Y-%m-%d') + '_dynamic.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df1.to_csv(filepath)
