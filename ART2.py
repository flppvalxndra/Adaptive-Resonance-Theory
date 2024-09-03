# Реализация нейронной сети ART2
# Feature recognition using ART2: A self-organizing neural network
# Kishore N Lankalapalli, Subrata Chatterjee and T.C. CHANG
# Первый (входной) слой n нейронов - слой сравнения F1
# Второй (выходной) слой m нейронов - слой распознавания F2
# Вектор входного сигнала x
# Вектор выходных сигналов y

import numpy as np


class ART2:

    def __init__(self, n, m=2, rho=0.8, theta=0.01, alpha=0.2,
                 a=10, b=10, c=0.1, d=0.9):
        """
        n - Number of nodes in F1 layer / Размер входного образа
        m - Number of clusters / Максимальное кол-во классов
        rho - Vigilance parameter / Порог чувствительности / Параметр бдительности
        theta - The noise suppression parameter / Параметр подавления шума
        alpha - Learning rate / Скорость обучения
        Веса t_ij (сверху вниз)
        Веса b_ji (снизу вверх)
        active - Количество активных классов
        """
        # Step 0 Initialize parameters
        self.a = a
        self.b = b
        self.theta = theta
        self.c = c
        self.d = d
        self.alpha = alpha
        self.rho = rho
        self.weight_t = np.full((n, m), 0.0)
        self.weight_b = np.full((m, n), 1.0/((1-d)*np.sqrt(n)))
        self.active = []

    @staticmethod
    def N(x):   # Normalize(x)
        return x / np.sqrt((x * x).sum())

    def T(self, x):   # Threshold(x) Пороговая функция для подавления шумовых сигналов
        for i in range(len(x)):
            if self.theta >= x[i] > 0:
                # x[i] = (2 * self.theta * x[i] * x[i]) / (x[i] * x[i] + self.theta * self.theta)
                x[i] = 0.0
        return x

    # Step 1
    # Do steps 2-12 N_EP (number of epochs) times.
    # An epoch is one presentation of each pattern.
    # Step 2
    # Do steps 3=11 for each input vector s.
    def learn(self, s):
        x = ART2.N(s)
        v = ART2.T(self, x)

        u = ART2.N(v)
        w = s + self.a*u
        p = u
        x = ART2.N(w)
        q = ART2.N(p)
        v = ART2.T(self, x) + self.b*ART2.T(self, q)
        # Step 4
        # Compute the net input of the F2 units.
        y = np.dot(self.weight_b, p)  # 6
        # Step 5-6
        act = len(self.active)
        num_class = np.argsort(-y[:act])  # Номера наиболее подходящих классов
        #print('Номера подходящих классов:', num_class)
        # Step 7
        for i in num_class:
            u = ART2.N(v)
            p = u + self.d * self.weight_t[:, i]
            r = (u + self.c*p) / (np.sqrt((u*u).sum()) + self.c * np.sqrt((p*p).sum()))
            similarity = np.sqrt((r*r).sum())
            #print('Коэффициент подобия: ',  str(round(similarity,6)))
            if np.sqrt((r*r).sum()) >= self.rho:
                w = s + self.a * u
                x = ART2.N(w)
                q = ART2.N(p)
                v = ART2.T(self, x) + self.b * ART2.T(self, q)
                # Step 9
                self.weight_t[:, i] = self.alpha*self.d*u + (1 + self.alpha*self.d*(self.d-1))*self.weight_t[:, i]
                self.weight_b[i, :] = self.alpha*self.d*u + (1 + self.alpha*self.d*(self.d-1))*self.weight_b[i, :]
                # Step 10 Update F1
                u = ART2.N(v)
                w = s + self.a * u
                p = u + self.d * self.weight_t[:, i]
                x = ART2.N(w)
                q = ART2.N(p)
                v = ART2.T(self, x) + self.b * ART2.T(self, q)
                # Дальше не используется???
                #print("Класс:", i, '\n')
                return "Класс: ", i, str('Коэффициент подобия: ' + str(round(similarity,6)))

        if act < y.size:
            i = act
            self.weight_t[:, i] = self.alpha*self.d*u + (1 + self.alpha * self.d * (self.d - 1)) * self.weight_t[:,i]
            self.weight_b[i, :] = self.alpha*self.d*u + (1 + self.alpha * self.d * (self.d - 1)) * self.weight_b[i,:]
            # Step 10 Update F1
            u = ART2.N(v)
            w = s + self.a * u
            p = u + self.d * self.weight_t[:, i]
            x = ART2.N(w)
            q = ART2.N(p)
            v = ART2.T(self, x) + self.b * ART2.T(self, q)
            self.active += [i]
            #print("Новый класс:", i, '\n')
            return str("Новый класс : "), i, 1
        else:
            #print("Класс не определен")
            return None



