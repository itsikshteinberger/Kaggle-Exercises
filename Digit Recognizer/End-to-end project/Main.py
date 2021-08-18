from tkinter import YES, BOTH
import PIL
import numpy as np
from PIL import ImageDraw, ImageOps
from Model import Net
import torch
import tkinter as tk
from matplotlib.pyplot import imshow
from tkinter import *
import PIL
from PIL import Image, ImageDraw


def predict():
    Cur_img = image1
    Cur_img = ImageOps.grayscale(Cur_img)
    Cur_img = Cur_img.resize((28, 28), Image.ANTIALIAS)
    Cur_img = np.array(Cur_img)
    tensor = torch.from_numpy(Cur_img)
    tensor = tensor.reshape(1, 1, 28, 28)
    
    with torch.no_grad():
        output = net(tensor.float())
        my_string_var.set("Your digit is " + str(torch.argmax(output).item()))


def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def clear():
    image1 = PIL.Image.new('RGB', (28 * 10, 28 * 10), 'black')
    draw.rectangle((0, 0, 280, 280), fill='black')
    cv.delete("all")



def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=width_scale.get(), fill='white')
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='white', width=width_scale.get())
    lastx, lasty = x, y


net = Net()
net.load_state_dict(torch.load('./model.pth'))


width = 20
root = Tk()
root.title("Digit Recognizer")
lastx, lasty = None, None
image_number = 0

my_string_var = tk.StringVar(value="")

cv = Canvas(root, width=28 * 10, height=28 * 10, bg='black')
# --- PIL
image1 = PIL.Image.new('RGB', (28 * 10, 28 * 10), 'black')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_predict = Button(text="Predict", command=predict)
btn_clear = Button(text="Clear", command=clear)
predict_label = Label(textvariable=my_string_var)
width_scale = Scale(orient=HORIZONTAL)

width_scale.pack(side=BOTTOM)
btn_predict.pack(side=LEFT)
btn_clear.pack(side=LEFT)

predict_label.pack(side=RIGHT)

root.mainloop()
