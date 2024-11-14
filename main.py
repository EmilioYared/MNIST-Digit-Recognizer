import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pandas as pd

W1 = np.array(pd.read_csv("newW1.csv", header=None))
W2 = np.array(pd.read_csv("newW2.csv", header=None))
W3 = np.array(pd.read_csv("newW3.csv", header=None))
b1 = np.array(pd.read_csv("newb1.csv", header=None)).T
b2 = np.array(pd.read_csv("newb2.csv", header=None)).T
b3 = np.array(pd.read_csv("newb3.csv", header=None)).T


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def get_predictions(A3):
    #returns the index of the max number inside the predicted output layer
    return np.argmax(A3, 0)
def make_predictions(X, W1, b1, W2, b2, W3 , b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    #makes the forward prop and get the final layer
    predictions = get_predictions(A3)
    return predictions

def predict_number():
    #opening the image and coverting it to grayscale
    imagex = Image.open('number.png').convert('L')
    matrix = np.array(np.array(imagex))
    #flatten makes it such that its a 1dimensional vector
    X = matrix.flatten()
    #we reshape so that it is a column vector
    X = X.reshape(-1, 1)
    X = X / 255
    print(X)
    return make_predictions(X,W1,b1,W2,b2,W3,b3)
class DrawNumberApp:
    def __init__(self, root):
        self.root = root
        #sets the title of our window
        self.root.title("Draw a Number")

        #////////
        # canvas part of the window
        self.canvas_width = 280
        self.canvas_height = 280
        self.brush_size = 8
        self.color = "black"
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        #we put the canvas at the top of our window works like a divsion
        self.canvas.pack()
        #/////////


        #/////////
        #we created a frame of buttons inside our window this will be our buttons section
        self.button_frame = tk.Frame(root)
        #this frame will be just after our canvas
        self.button_frame.pack()
        #1st button that allows us to clear the screen it calls clear canvas method
        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side="left") #we specified the side
        #save the image before predicting
        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_image)
        self.save_button.pack(side="left")
        #2nd button that allows us to predict the number drawn on the canvas it calls predict method
        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side="left")
        #/////////


        #we binded the left mouse butten to the paint method
        self.canvas.bind("<B1-Motion>", self.paint)

        #creation of an image grayscale one with the size of the canvas and statring values of 255
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)

        #this will allow us to draw on the image
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        #coordinates for a rectangle centered at the mouse position
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)

        #draws an oval in this case a dot because x1,y1,x2,y2 symmetric (on canvas)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color, outline=self.color)

        #we need to also draw on the image we replicate whatever is drawn here (on image)
        self.draw.ellipse([x1, y1, x2, y2], fill=self.color)

    def clear_canvas(self):
        #clear the canvas
        self.canvas.delete("all")
        #and we also clear the image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

    def save_image(self):
        #we resize to 28x28
        small_image = self.image.resize((28, 28))
        #invert image colors
        small_image = ImageOps.invert(small_image)
        #we save the image
        small_image.save("number.png")
        print("Image saved as number.png")

    def predict(self):
        #get the prediction from the model
        predicted_number = predict_number()
        #display the prediction in a popup window
        messagebox.showinfo("Prediction", f"The predicted number is: {predicted_number}")

def main():
    #this creates an empty window
    root = tk.Tk()
    #object creation to make it compact and add functionality
    app = DrawNumberApp(root)

    root.mainloop()



main()