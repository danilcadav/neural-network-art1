import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import TeachedART as nn
import numpy as np
import os 

Pathico = os.getcwd() + '\icon.png'

result = []
concur = []
numbers = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']

def open_img():
    x = filedialog.askopenfilename(title ='Загрузите фото')
    img = Image.open(x)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_array = np.asarray(img.convert('L'))
    img = img.resize((300, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(win, image = img)
    panel.image = img
    panel.place(x = 100, y = 100)
    img_data = 1 - (img_array.reshape(784)/255.0)
    img_data = np.around(img_data,0)
    Concurrence = nn.giveResult(img_data)
    global result 
    global concur
    result.append(str(np.argmax(Concurrence)))
    concur.append(Concurrence)
    label_1 = tk.Label(win, text = '{}'.format(np.argmax(Concurrence)), bg = '#333', fg = '#FFF', font = ('Arial', 15,'bold'))
    label_1.place(x = 500, y = 240)
    
def open_outputfile():
    x = filedialog.askopenfilename(title ='Выберите файл для записи')
    f = open(x, "w")
    for i in range(0, len(result)):
        f.write("Результат {}-го образа: ".format(i + 1))
        f.write(result[i])
        f.write("\n")
        f.writelines("     Сходство с %s: %s%%\n" %(k, s) for k, s in zip(numbers, np.around(concur[i], 1)))
        f.write("\n")
        label_1 = tk.Label(win, text = 'Записано!', bg = '#333', fg = '#FFF', font = ('Arial', 15,'bold'))
        label_1.place(x = 500, y = 350)

win = tk.Tk()
win.title('Neural Network ART-1')
photo = tk.PhotoImage(file = Pathico)
win.iconphoto(False, photo)
win.config(bg = '#333')
win.geometry("800x600")
win.resizable(False,False)

btn = tk.Button(win, text ='Выбрать изображение', command = open_img, width = 30)
btn.place(x = 500, y = 150)

btn = tk.Button(win, text ='Выбрать файл для записи', command = open_outputfile, width = 30)
btn.place(x = 500, y = 300)

canvas = tk.Canvas(win, width = 310, height = 310, bg = '#444')
canvas.place(x = 95, y = 95)

label_1 = tk.Label(win, text = 'Загрузите фото:', bg = '#333', fg = '#FFF', font = ('Arial', 15,'bold'))
label_1.place(x = 500, y = 100)
label_1 = tk.Label(win, text = 'Результат:', bg = '#333', fg = '#FFF', font = ('Arial', 15,'bold'))
label_1.place(x = 500, y = 200)
win.mainloop()
