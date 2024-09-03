# Реализация алгоритма из книги "Методы и технологии ИИ" Рутковский Л.
# Первый (входной) слой n нейронов - слой сравнения F1
# Второй (выходной) слой m нейронов - слой распознавания F2
# Вектор входного сигнала X
# Вектор выходных сигналов Y

import numpy as np


class ART1:

    def __init__(self, n, m=10, rho=0.87, l=2):
        """
        n - Размер входного образа
        m - Максимальное кол-во классов
        rho - Порог чувствительности / Параметр бдительности #0,82 -> b, d в одном классе
        l - Скорость Обучения (Константа > 1)
        Веса t_ij (сверху вниз)
        Веса b_ji (снизу вверх)
        active - Количество активных классов
        """
        # Шаг 1.
        # Начальные значения весов
        self.weight_t = np.full((n, m), 1)
        self.weight_b = np.full((m, n), l/(l-1+n))
        self.rho = rho
        self.active = []

    def learn(self, x):
        # Шаг 2.
        # Входной вектор X (значения 0 или 1)
        act = len(self.active)
        f1 = np.dot(self.weight_b, x)  # Меры соответствия входного образа
        num_class = np.argsort(-f1[:act])   # Номера наиболее подходящих классов
        # print('Номера подходящих классов:', num_class)
        # Шаг 3.
        for i in num_class:
            # Проверка выполнения условия
            d = (self.weight_t[:, i] * x).sum() / x.sum()
            #print('Коэффициент подобия:', d)
            if d > self.rho:

                # Шаг 4 Модификация весов
                self.weight_t[:, i] *= x
                self.weight_b[i, :] = self.weight_t[:, i] / (0.5 + (self.weight_t[:, i] * x).sum())
                #print("Класс:", i, '\n')
                return str("Класс: "), i , '\nКоэффициент подобия: '+str(round(d,6))
        if act < f1.size:
            i = act
            self.weight_t[:, i] *= x
            self.weight_b[i, :] = self.weight_t[:, i] / (0.5 + (self.weight_t[:, i] * x).sum())
            self.active += [i]
            #print("Новый класс:", i, '\n')
            return str("Новый класс: "), i, 1
        else:
            return None
