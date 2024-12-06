from ART1 import ART1
from ART2 import ART2
from numpy import array, genfromtxt, unique
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.messagebox import showerror, showinfo
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import subprocess


class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

    def show(self):
        self.lift()


class Page1(Page):

    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        self.network = None
        self.filepath = None
        self.data = None
        self.width = 0
        self.height = 0

        label0 = tk.Label(self, text="Нейронная сеть ART1", font=("Arial", 10, "bold"))
        label0.grid(column=0, row=0, columnspan=1, ipady=5, sticky="NSEW")

        self.frame1 = tk.LabelFrame(self, text="Параметры нейронной сети", pady=5)
        self.frame1.grid(column=0, row=1, sticky="NW")

        label1 = tk.Label(self.frame1, text="Введите кол-во эпох:", width=27, anchor="nw")
        label1.grid(column=0, row=1, padx=3, sticky="NSEW")
        epoch = tk.IntVar()
        epoch.set(2)
        self.entry1 = tk.Entry(self.frame1, width=3, textvariable=epoch)
        self.entry1.grid(column=1, row=1, padx=5, sticky="EW")

        label2 = tk.Label(self.frame1, text="Размер входного образа n:", width=27, anchor="nw")
        label2.grid(column=0, row=2, padx=3)
        self.entry2 = tk.Entry(self.frame1, width=5, textvariable=tk.IntVar(), state="disabled")
        self.entry2.grid(column=1, row=2, padx=5, sticky="EW")

        label3 = tk.Label(self.frame1, text="Максимальное кол-во классов m:", width=27, anchor="nw")
        label3.grid(column=0, row=3, padx=3)
        m = tk.IntVar()
        m.set(10)
        self.entry3 = tk.Entry(self.frame1, width=3, textvariable=m)
        self.entry3.grid(column=1, row=3, padx=5, sticky="EW")

        label4 = tk.Label(self.frame1, text="Параметр бдительности ρ,\nгде 0 < ρ < 1:",
                          width=27, anchor="w", justify="left")
        label4.grid(column=0, row=4, padx=3, sticky="NSEW")
        rho = tk.DoubleVar()
        rho.set(0.79)
        self.entry4 = tk.Entry(self.frame1, width=3, textvariable=rho)
        self.entry4.grid(column=1, row=4, padx=5, sticky="EW")

        label5 = tk.Label(self.frame1, text="Константа L,\nгде L ∈ N:",
                          width=27, anchor="w", justify="left")
        label5.grid(column=0, row=5, padx=3, sticky="W")
        l = tk.IntVar()
        l.set(2)
        self.entry5 = tk.Entry(self.frame1, width=3, textvariable=l)
        self.entry5.grid(column=1, row=5, padx=5, sticky="EW")

        button1 = tk.Button(self.frame1, text='Выбрать файлы', command=self.open_file)
        button1.grid(padx=30, sticky="NSEW")
        button2 = tk.Button(self.frame1, text='Обучение нейросети', command=self.start)
        button2.grid(padx=30, sticky="NSEW")

        self.frame2 = tk.LabelFrame(self, text="Результаты кластеризации", pady=5)
        self.frame2.grid(column=1, row=1)
        self.canvas = tk.Canvas(self.frame2, bg='lightgrey', width=625, height=500)
        self.canvas.grid(row=0, column=0, sticky="nsew")


    def open_file(self):
        files = filedialog.askopenfiles(title="Выбор файла")
        if files:
            self.filepath = []
            for i in range(len(files)):
                self.filepath.append(files[i].name)
            with Image.open(self.filepath[0]) as img:
                self.width, self.height = img.size
            imgsize = tk.IntVar()
            imgsize.set(self.width * self.height)
            self.entry2["textvariable"] = imgsize

    def create_network(self):
        self.network = ART1(n=int(self.entry2.get()),
                            m=int(self.entry3.get()),
                            rho=float(self.entry4.get()),
                            l=int(self.entry5.get()))

    def showImage(self):
        self.data = []
        y = 50
        for i in range(len(self.filepath)):
            # Вывод изображения
            img = Image.open(self.filepath[i])
            self.image[0].append(ImageTk.PhotoImage(img))
            self.canvas.create_image(3, y, image=self.image[0][i], anchor='nw')
            y += self.height + 5
            # Перевод изображения в матрицу из 0 и 1
            gray = img.convert('L')
            bw = gray.point(lambda x: 1 if x < 128 else 0, '1')
            self.data.append(array(bw.getdata()))

    def learnPattern(self):
        epoch = int(self.entry1.get())
        self.canvas.create_text(5, 35, text='', anchor='nw')
        x = self.width + 10
        y = 50
        for k in range(int(self.entry3.get())):
            self.countClass['Класс '+str(k)] = ([0] * epoch)
        for i in range(epoch):
            res = []
            self.canvas.create_text(x, 35, text='Эпоха №'+str(i+1),
                                    font=("Arial", 8, "bold"), anchor='nw')
            for j in range(len(self.data)):
                res.append(self.network.learn(self.data[j]))
                if res[j] == None:
                    self.canvas.create_text(x, y, text='Класс не определен', anchor='nw', tag='text')
                elif res[j][2] == 1:
                    self.canvas.create_text(x, y, text=res[j][0]+str(res[j][1]), anchor='nw', tag='text')
                    self.countClass['Класс ' + str(res[j][1])][i] += 1
                else:
                    self.canvas.create_text(x, y, text=res[j][0] + str(res[j][1])+ res[j][2], anchor='nw')
                    self.countClass['Класс ' + str(res[j][1])][i] += 1
                y += self.height + 5
            x += 190
            y = 50

    def clean(self):
        self.canvas.delete("all")

    def start(self):
        if self.filepath is None:
            showerror(title="Ошибка",
                      message="Выберите файлы для обучения/распознавания")
        else:
            self.image = [[], []]  # Для вывода изображений
            self.countClass = {}  # {'Название класса':[сколько раз экземпляр был отнесен к классу]}
            self.canvas.delete("all")

            self.scrollbarX = ttk.Scrollbar(self.frame2, orient="horizontal", command=self.canvas.xview)
            self.scrollbarY = ttk.Scrollbar(self.frame2, orient="vertical", command=self.canvas.yview)
            self.scrollable_frame = ttk.Frame(self.canvas)
            self.scrollable_frame.bind("<Configure>",
                                       lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.canvas.configure(xscrollcommand=self.scrollbarX.set, yscrollcommand=self.scrollbarY.set)
            self.scrollbarX.grid(row=1, column=0, sticky="ew")
            self.scrollbarY.grid(row=0, column=1, sticky="ns")

            button3 = tk.Button(self.canvas, text='Очистить', command=self.clean)
            button3.place(x=5, y=5)
            button4 = tk.Button(self.canvas, text='Подробнее о результатах кластеризации', command=self.info)
            button4.place(x=75, y=5)
            self.create_network()
            self.showImage()
            self.learnPattern()

    def info(self):
        maxC = len(self.network.active)
        size = self.height * self.width
        img = []
        for i in range(0, maxC):
            img.append([])
            res = self.network.weight_t[:, i]
            for j in range(size):
                if self.network.weight_t[:, i][j] == 0:
                    res[j] = 255
                elif self.network.weight_t[:, i][j] == 1:
                    res[j] = 1
            img[i] = res.reshape(self.height, self.width)
        self.window = tk.Tk()
        self.window.title("Информация о результатах кластеризации")

        self.frame1 = tk.LabelFrame(self.window, text="Критические черты класса")
        self.frame1.grid(column=0, row=0, sticky="NSEW")

        self.canvas0 = tk.Canvas(self.frame1, width=450, height=450)
        self.canvas0.delete("all")

        label = tk.Label(self.frame1, text='Критические черты классов в нейронной сети ' +
                                           'представлены матрицей нисходящих весов,\n' +
                                           'где каждый столбец матрицы соответствует определенному классу.\n' +
                                           'Ниже для наглядности выведены критические черты каждого активного класса,\n'
                                           'в виде изображения, где белый цвет отражает 0, а черный - 1.\n' +
                                           'Также отображено количество образов, отнесенных к данному классу '
                                           'в соответствии с эпохами.', anchor="w", justify="left")
        label.grid(row=0, column=0, sticky="nsew", pady = 10)

        scrollbarY = ttk.Scrollbar(self.frame1, orient="vertical", command=self.canvas0.yview)
        scrollbarX = ttk.Scrollbar(self.frame1, orient="horizontal", command=self.canvas0.xview)
        scrollable_frame = ttk.Frame(self.canvas0)

        scrollable_frame.bind("<Configure>", lambda e: self.canvas0.configure(scrollregion=self.canvas0.bbox("all")))
        self.canvas0.create_window((0, 0), window=scrollable_frame, anchor="nw")
        self.canvas0.configure(yscrollcommand=scrollbarY.set, xscrollcommand=scrollbarX.set)
        self.canvas0.grid(row=1, column=0, sticky="nsew")
        scrollbarY.grid(row=1, column=1, sticky="ns")
        scrollbarX.grid(row=2, column=0, sticky="ew")
        self.image[1]=[]
        x = self.width + 20
        y = 7
        for i in range(int(self.entry1.get())):
            self.canvas0.create_text(x, y, text='Эпоха №' + str(i + 1), font=("Arial", 8, "bold"), anchor="nw")
            x += 200
        x = 10
        y = 20
        for i in range(maxC):
            image = Image.fromarray(img[i])
            self.image[1].append(ImageTk.PhotoImage(image=image, master= self.canvas0))
            self.canvas0.create_image(x, y, image=self.image[1][i], anchor="nw", )
            for j in range(int(self.entry1.get())):
                self.canvas0.create_text(self.width + x + 10, y, text = 'Класс '+ str(i),
                                         anchor="nw", justify="left")
                self.canvas0.create_text(self.width + x + 10, y+13, text='Количество образов, \n' +
                                         'отнесенных к данному классу: ' + str(self.countClass['Класс ' + str(i)][j]),
                                         anchor="nw", justify="left")
                x += 200
            x = 10
            y += self.height + 20


class Page2(Page):
    def __init__(self, *args, **kwargs):
        Page.__init__(self, *args, **kwargs)
        self.network = None
        self.data = []
        self.nameClass = None    # Какому классу соответствует элемент согласно данным файла (по индексу)
        self.uniqueClass = None  # Уникальные названия классов
        self.numClass = {}     # {'Название класса':[общее количество экземпляров класса]}
        self.countClass = {}     # {'Название класса':[сколько раз экземпляр был отнесен к классу]}

        w = 28
        label0 = tk.Label(self, text="Нейронная сеть ART2", font=("Arial", 10, "bold"))
        label0.grid(column=0, row=0, columnspan=1, ipady=5, sticky="NSEW")

        self.frame1 = tk.LabelFrame(self, text="Параметры нейронной сети", pady=5)
        self.frame1.grid(column=0, row=1, sticky="NSEW")

        label1 = tk.Label(self.frame1, text="Введите кол-во эпох:", width=w, anchor="nw")
        label1.grid(column=0, row=1, padx=3, sticky="NSEW")
        epoch = tk.IntVar()
        epoch.set(2)
        self.entry1 = tk.Entry(self.frame1, width=6, textvariable=epoch)
        self.entry1.grid(column=1, row=1, padx=5, sticky="EW")

        label2 = tk.Label(self.frame1, text="Размер входного образа n:", width=w, anchor="nw")
        label2.grid(column=0, row=2, padx=3)
        self.entry2 = tk.Entry(self.frame1, width=7, textvariable=tk.IntVar(), state="disabled")
        self.entry2.grid(column=1, row=2, padx=5, sticky="EW")

        label3 = tk.Label(self.frame1, text="Максимальное кол-во классов m:", width=w, anchor="nw")
        label3.grid(column=0, row=3, padx=3)
        m = tk.IntVar()
        m.set(3)
        self.entry3 = tk.Entry(self.frame1, width=7, textvariable=m)
        self.entry3.grid(column=1, row=3, padx=5, sticky="EW")

        label4 = tk.Label(self.frame1, text="Параметр бдительности ρ,\n"
                                            "где 0 < ρ < 1:", width=w, anchor="w", justify="left")
        label4.grid(column=0, row=4, padx=3, sticky="NSEW")
        rho = tk.DoubleVar()
        rho.set(0.9995)
        self.entry4 = tk.Entry(self.frame1, width=7, textvariable=rho)
        self.entry4.grid(column=1, row=4, padx=5, sticky="EW")

        label5 = tk.Label(self.frame1, text="Параметр подавления шума θ,\n"
                                            "где 0 < θ < 1:", width=w, anchor="w", justify="left")
        label5.grid(column=0, row=5, padx=3, sticky="W")
        l = tk.DoubleVar()
        l.set(0.01)
        self.entry5 = tk.Entry(self.frame1, width=7, textvariable=l)
        self.entry5.grid(column=1, row=5, padx=5, sticky="EW")

        label6 = tk.Label(self.frame1, text="Скорость обучения α,\nгде 0 < α < 1:",
                          width=w, anchor="w", justify="left")
        label6.grid(column=0, row=6, padx=3, sticky="W")
        l = tk.DoubleVar()
        l.set(0.15)
        self.entry6 = tk.Entry(self.frame1, width=7, textvariable=l)
        self.entry6.grid(column=1, row=6, padx=5, sticky="EW")

        labelA = tk.Label(self.frame1, text="Фиксированный вес слоя F1 - a:",
                          width=w, anchor="w", justify="left")
        labelA.grid(column=0, row=7, padx=3, sticky="W")
        l = tk.IntVar()
        l.set(10)
        self.entryA = tk.Entry(self.frame1, width=7, textvariable=l)
        self.entryA.grid(column=1, row=7, padx=5, sticky="EW")

        labelB = tk.Label(self.frame1, text="Фиксированный вес слоя F1 - b:",
                          width=w, anchor="w", justify="left")
        labelB.grid(column=0, row=8, padx=3, sticky="W")
        l = tk.IntVar()
        l.set(10)
        self.entryB = tk.Entry(self.frame1, width=7, textvariable=l)
        self.entryB.grid(column=1, row=8, padx=5, sticky="EW")

        labelC = tk.Label(self.frame1, text="Фиксированный вес c:",
                          width=w, anchor="w", justify="left")
        labelC.grid(column=0, row=9, padx=3, sticky="W")
        l = tk.DoubleVar()
        l.set(0.1)
        self.entryC = tk.Entry(self.frame1, width=7, textvariable=l)
        self.entryC.grid(column=1, row=9, padx=5, sticky="EW")

        labelD = tk.Label(self.frame1, text="Фиксированный вес d:",
                          width=w, anchor="w", justify="left")
        labelD.grid(column=0, row=10, padx=3, sticky="W")
        l = tk.DoubleVar()
        l.set(0.9)
        self.entryD = tk.Entry(self.frame1, width=7, textvariable=l)
        self.entryD.grid(column=1, row=10, padx=5, sticky="EW")

        # ------------------------------------------------------------------------------------

        self.frame2 = tk.LabelFrame(self, text="Настройка данных для обучения", pady=5)
        self.frame2.grid(column=0, row=2, sticky="NW")

        label5 = tk.Label(self.frame2, text="Строка, исп. для разделения значений:", anchor="w", justify="left")
        label5.grid(column=0, row=0, columnspan=4, sticky="W", padx=3)
        delimiter = tk.StringVar()
        delimiter.set(',')
        self.delimiter = tk.Entry(self.frame2, width=3, textvariable=delimiter)
        self.delimiter.grid(column=3, row=0, padx=5, sticky="NSEW")

        label0 = tk.Label(self.frame2, text="Количество строк, \nкоторые нужно пропустить:",width=23,
                          anchor="w", justify="left")
        label0.grid(column=0, row=1, padx=3, sticky="W", columnspan=6, rowspan=1)

        label1 = tk.Label(self.frame2, text="в начале файла -")
        label1.grid(column=0, row=2,padx=3)
        skip_header = tk.IntVar()
        skip_header.set(1)
        self.skip_header = tk.Entry(self.frame2, textvariable=skip_header, width=3)
        self.skip_header.grid(column=1, row=2)

        label2 = tk.Label(self.frame2, text="в конце файла -")
        label2.grid(column=2, row=2, sticky="EW")
        skip_footer = tk.IntVar()
        skip_footer.set(0)
        self.skip_footer = tk.Entry(self.frame2, width=3, textvariable=skip_footer)
        self.skip_footer.grid(column=3, row=2, padx=5, sticky="NSEW")

        label3 = tk.Label(self.frame2, text="Какие столбцы следует считывать,\nнумерация c 0:",
                          width=27, anchor="w", justify="left")
        label3.grid(column=0, row=3, padx=3, sticky="W", columnspan=6, rowspan=1)
        label4 = tk.Label(self.frame2, text="от -", width=14, anchor="se", justify="right")
        label4.grid(column=0, row=4, padx=3, sticky="W")
        Colstart = tk.IntVar()
        Colstart.set(1)
        self.Colstart = tk.Entry(self.frame2, width=3, textvariable=Colstart,  justify="left")
        self.Colstart.grid(column=1, row=4, sticky="nsew")
        label5 = tk.Label(self.frame2, text="до -", width=14, anchor="se", justify="right")
        label5.grid(column=2, row=4, sticky="EW")
        Colstop = tk.IntVar()
        Colstop.set(5)
        self.Colstop = tk.Entry(self.frame2, width=3, textvariable=Colstop)
        self.Colstop.grid(column=3, row=4, padx=5, sticky="NSEW")

        label6 = tk.Label(self.frame2, text="В каком столбце находится\nинформация о классе:",
                          width=23, anchor="w", justify="left")
        label6.grid(column=0, row=5, padx=3, sticky="W", columnspan=6, rowspan=1)
        colName = tk.IntVar()
        colName.set(-1)
        self.colName = tk.Entry(self.frame2, width=3, textvariable=colName, justify="left")
        self.colName.grid(column=3, row=5, padx=5, sticky="ew")

        button1 = tk.Button(self.frame2, text='Просмотр файла', command=self.view_file)
        button1.grid(column=0, row=6, padx=10, pady=5, columnspan=2, rowspan=1,)

        button1 = tk.Button(self.frame2, text='Выбрать файл .csv', command=self.open_file)
        button1.grid(column=2, row=6, padx=10, pady=5, columnspan=2, rowspan=1, )

        button2 = tk.Button(self.frame2, text='Обучение нейросети', command=self.start)
        button2.grid(padx=10, row=8, sticky="NSEW", columnspan=6)

        self.frame3 = tk.LabelFrame(self, text="Результаты кластеризации", pady=5)
        self.frame3.grid(column=1, row=1, rowspan=2)
        self.canvas = tk.Canvas(self.frame3, bg='lightgrey', width=600, height=496)
        self.canvas.grid(row=0, column=0, sticky="nsew")


    def view_file(self):
        file = filedialog.askopenfilename()
        subprocess.Popen((['C:\\Windows\\System32\\notepad.exe', file]))
    def open_file(self):
        file = filedialog.askopenfilename(title="Выбор файла")
        if file:
            skip_header = int(self.skip_header.get())
            skip_footer = int(self.skip_footer.get())
            start = int(self.Colstart.get())
            stop = int(self.Colstop.get())
            delimiter = str(self.delimiter.get())
            colName = int(self.colName.get())

            self.data = genfromtxt(fname=file, delimiter=delimiter, dtype=None,
                                   skip_header=skip_header, skip_footer=skip_footer,
                                   usecols=range(start, stop), encoding=None)
            self.nameClass = genfromtxt(fname=file, delimiter=delimiter, dtype=str,
                                        skip_header=skip_header, skip_footer=skip_footer,
                                        usecols=colName, encoding=None)
            self.uniqueClass = unique(self.nameClass)
            datasize = tk.IntVar()
            datasize.set(int(self.data.shape[1]))
            self.entry2["textvariable"] = datasize

    def create_network(self):
        self.network = ART2(n=int(self.entry2.get()),
                            m=int(self.entry3.get()),
                            rho=float(self.entry4.get()),
                            theta=float(self.entry5.get()),
                            alpha=float(self.entry6.get()),
                            a=int(self.entryA.get()),
                            b=int(self.entryB.get()),
                            c=float(self.entryC.get()),
                            d=float(self.entryD.get()))

    def start(self):
        if len(self.data):
            self.canvas.delete("all")
            self.scrollbarX = ttk.Scrollbar(self.frame3, orient="horizontal", command=self.canvas.xview)
            self.scrollbarY = ttk.Scrollbar(self.frame3, orient="vertical", command=self.canvas.yview)
            self.scrollable_frame = ttk.Frame(self.canvas)
            self.scrollable_frame.bind("<Configure>",
                                       lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            self.canvas.configure(xscrollcommand=self.scrollbarX.set, yscrollcommand=self.scrollbarY.set)
            self.scrollbarX.grid(row=1, column=0, sticky="ew")
            self.scrollbarY.grid(row=0, column=1, sticky="ns")
            self.create_network()
            self.learnPattern()
        else:
            showerror(title="Ошибка",
                      message="Выберите файлы для обучения/распознавания")

    def learnPattern(self):
        button3 = tk.Button(self.canvas, text='Подробнее о результатах кластеризации',  command=self.info)
        button3.place(x=75, y=5)
        buttonC = tk.Button(self.canvas, text='Очистить', command=self.clean)
        buttonC.place(x=5, y=5)

        epoch = int(self.entry1.get())
        x = 25
        y = 50
        self.canvas.create_text(3, 35, text='№',font=("Arial", 8, "bold"), anchor='nw')
        self.canvas.create_text(x, 35, text='Данные:', font=("Arial", 8, "bold"), anchor='nw')

        for j in range(len(self.data)):
            self.canvas.create_text(3, y, text=str(j), anchor='nw')
            self.canvas.create_text(x, y, text=str(self.data[j]), anchor='nw', width=80)
            y += 50
        for i in range(len(self.uniqueClass)):
            self.countClass[str(self.uniqueClass[i])] = []
            self.numClass[str(self.uniqueClass[i])] = []
        for i in range(0, epoch):
            res = []
            self.canvas.create_text(x+87, 35, text='Эпоха №' + str(i+1),
                                    font=("Arial", 8, "bold"), anchor='nw')
            y = 50
            for k in range(len(self.uniqueClass)):
                self.countClass[str(self.uniqueClass[k])].append([0] * int(self.entry3.get()))
                self.numClass[str(self.uniqueClass[k])].append(0)
            for j in range(len(self.data)):
                res.append(self.network.learn(self.data[j]))
                if res[j] == None:
                    self.canvas.create_text(x+87, y, text="Класс не определен" + "\nКласс, согласно файлу: " +
                                            str(self.nameClass[j]), anchor='nw')
                    self.numClass[self.nameClass[j]][i] += 1
                elif res[j][2] == 1:
                    self.canvas.create_text(x+87, y, text=res[j][0] + str(res[j][1]) +
                                            "\nКласс, согласно файлу: " + str(self.nameClass[j]), anchor='nw')

                    self.countClass[self.nameClass[j]][i][res[j][1]] += 1
                    self.numClass[self.nameClass[j]][i] += 1
                else:
                    self.canvas.create_text(x+87, y, text=res[j][0] + str(res[j][1])+'\n' + res[j][2] +
                                            "\nКласс, согласно файлу: " + str(self.nameClass[j]), anchor='nw')
                    self.countClass[self.nameClass[j]][i][res[j][1]] += 1
                    self.numClass[self.nameClass[j]][i] += 1
                y += 50
            x += 210

    def clean(self):
        self.canvas.delete("all")

    def info(self):
        window = tk.Tk()
        window.title("Информация о результатах кластеризации")
        window.geometry("951x614")

        frame1 = tk.LabelFrame(window, text="Графики весов нейронной сети", pady=5)
        frame1.grid(column=0, row=0, rowspan=2, columnspan=2, sticky="NSEW")

        label = tk.Label(frame1, text='Ниже представлена матрица нисходящих весов в виде графиков,\n' +
                                      'где каждый график отражает соответствующий  столбец матрицы,\n' +
                                      'который в свою очередь представляет определенный класс.\n' +
                                      'По оси Ox расположены индексы элементов,\n' +
                                      'по оси Oy их значения соответственно.\n',
                         font=("Arial", 9, "bold"), anchor="w", justify="left")
        label.pack()

        fig = Figure(figsize=(5, 5), dpi=70)
        plot1 = fig.add_subplot()
        plot1.plot(self.network.weight_t, marker = 'o')
        name = []
        for i in range(len(self.network.active)):
            name.append('Класс ' + str(i))
        fig.legend(name, loc=1)
        plot1.set(xlabel='Индекс параметра класса', ylabel='Значение параметра класса')

        canvas = FigureCanvasTkAgg(fig, master=frame1)
        canvas.draw()
        canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(canvas, frame1)
        toolbar.update()
        canvas.get_tk_widget().pack()

        frame = tk.LabelFrame(window, text="Соответствие результатов кластеризации предоставленным данным", pady=5)
        frame.grid(column=3, row=0, rowspan=2, columnspan=2, sticky="NSEW")

        canvas = tk.Canvas(frame, width=530, height=565)
        scrollbarY = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbarX = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbarY.set,xscrollcommand=scrollbarX.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbarY.grid(row=0, column=1, sticky="ns")
        scrollbarX.grid(row=1, column=0, sticky="ew")

        for i in range(int(self.entry1.get())):
            x = 0
            label = tk.Label(scrollable_frame, text='Эпоха №' + str(i+1),
                              font=("Arial", 9, "bold"), anchor="w", justify="left")
            label.grid(column=i, row=0, padx=3, sticky="W")
            size = int(self.entry3.get())
            for j in range(len(self.uniqueClass)):
                x += 1
                label1 = tk.Label(scrollable_frame, text='\nКласс: ' + self.uniqueClass[j],
                                  font=("Arial", 8, "bold"), anchor="w", justify="left")
                label1.grid(column=i, row=x, sticky="W")

                x += 1
                label2 = tk.Label(scrollable_frame, text='Количество экземпляров класса\n'
                                  + 'в обучающем наборе данных: ' + str(self.numClass[self.uniqueClass[j]][i]),
                                  anchor="w", justify="left")
                label2.grid(column=i, row=x, sticky="W")

                x += 1
                label3 = tk.Label(scrollable_frame, text='Сколько векторов, \nпринадлежащих классу выше,\n' +
                                                         'сеть отнесла к каждому из классов  ', anchor="w",
                                  justify="left")
                label3.grid(column=i, row=x, sticky="W",)

                for c in range(size):
                    x += 1
                    label3 = tk.Label(scrollable_frame, text='К классу ' + str(c) + ': ' +
                                         str(self.countClass[self.uniqueClass[j]][i][c]),
                                      anchor="w", justify="left")
                    label3.grid(column=i, row=x, sticky="W")

# ==============================================================================================================


class MainView(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        p1 = Page1(self)
        p2 = Page2(self)

        buttonframe = tk.Frame(self)
        container = tk.Frame(self)
        buttonframe.pack(side="top", fill="x", expand=False)
        container.pack(side="top", fill="both", expand=True)

        p1.place(in_=container, x=0, y=0, relwidth=1, relheight=1)
        p2.place(in_=container, x=0, y=0, relwidth=1, relheight=1)

        b1 = tk.Button(buttonframe, text="ART1", command=p1.lift)
        b2 = tk.Button(buttonframe, text="ART2", command=p2.lift)

        b1.pack(side="left")
        b2.pack(side="left")

        p1.show()

def infoART1():
    showinfo("Нейронная сеть ART1", "Нейронная сеть ART1 предназначена для обработки бинарных образов.\n" +
                                    "Данное приложение предоставляет возможность кластеризовать двоичные образы, " +
                                    "представленные в виде изображений в формате .bmp или .jpg.\n\n" +
                                    "Для работы необходимо: \n" +
                                    "1) Заполнить поля с желаемыми параметрами сети в окне " +
                                    "'Параметры нейронной сети'.\n" +
                                    "2) Выбрать нужные изображения по кнопке 'Выбрать файлы'.\n" +
                                    "3) Нажать кнопку 'Обучение нейросети'.\n\n" +
                                    "В результате в поле 'Результаты кластеризации' " +
                                    "будут представлены входные образы и результаты " +
                                    "их кластеризации согласно эпохам.\n" +
                                    "Также данное поле можно очистить с помощью кнопки 'Очистить'.\n" +
                                    "Подробнее о результатах можно узнать по кнопке " +
                                    "'Подробнее о результатах'.\n" +
                                    "При повторном изменении параметров нейронной сети, " +
                                    "будет создана новая сеть и обучение начнется заново.")

def infoART2():
    showinfo("Нейронная сеть ART2", "Нейронная сеть ART2 предназначена для обработки вещественных образов.\n" +
                                    "Данное приложение предоставляет возможность кластеризовать вещественные образы, " +
                                    "представленные в виде табличных данных в формате .csv.\n\n" +
                                    "Для работы необходимо: \n" +
                                    "1) Заполнить поля с желаемыми параметрами сети в окне " +
                                    "'Параметры нейронной сети'.\n" +
                                    "2) Заполнять поля в окне 'Настройка данных для обучения' " +
                                    "для корректного считывания данных.\n" +
                                    "Данные в виде файла можно заранее просмотреть с помощью кнопки 'Просмотр файла'.\n" +
                                    "3) Нажать кнопку 'Обучение нейросети'.\n\n" +
                                    "В результате в поле 'Результаты кластеризации' " +
                                    "будут представлены входные образы и результаты " +
                                    "их кластеризации по эпохам.\n" +
                                    "Также данное поле можно очистить с помощью кнопки 'Очистить'.\n" +
                                    "Подробнее о результатах можно узнать по кнопке " +
                                    "'Подробнее о результатах'.\n" +
                                    "При повторном изменении параметров нейронной сети, " +
                                    "будет создана новая сеть и обучение начнется заново.")

def infoReferences():
    showinfo("Литература", "Нейронные сети адаптивной резонансной теории ART1 и ART2 были предложены "
                           "Стивеном Гроссбергом и Гейлом Карпентером в 1987 году.\n\n"
                           "Статья посвященная сети ART1: Carpenter G. A., Grossberg S., "
                           "“A massively parallel architecture for a self-organizing "
                           "neural pattern recognition machine”.\n"
                           "Статья посвященная сети ART2:  Carpenter G. A., Grossberg S., "
                           "“ART 2: self-organization of stable category recognition "
                           "codes for analog input patterns”.\n\n"
                           "Алгоритм работы сети ART1 был взят из книги: Рутковский Л. "
                           "“Методы и технологии искусственного интеллекта“.\n"
                           "Алгоритм работы сети ART2 был взят из книги: Laurene V. Fausett "
                           "“Fundamentals of Neural Networks: Architectures, Algorithms And Applications“.\n")

def infoAutor():
    showinfo("Авторские права","Приложение разработано для демонстрации результатов работы нейронных"
                               " сетей теории адаптивного резонанса: \n\n"
                               "ART1 - для обработки бинарных образов,\n" 
                               "ART2 - для обработки вещественных образов.\n" 
                           "\nАвтор: Филиппова Александра Андреевна.")

