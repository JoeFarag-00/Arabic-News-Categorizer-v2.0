import customtkinter
import os
from tkinter import messagebox
from tkinter import filedialog
customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('green')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from tkinter import Tk, Label, Text, Button, Menu

class MainGUI():
    
    @staticmethod
    def Preprocess_Text(text):
        global w2v_model
        global model
        if Selected == 1:
            model = load_model('Models/cnn_model.h5')
        elif Selected == 2:
            model = load_model('Models/SVM_model.h5')
        elif Selected == 3:
            model = load_model('Models/cnn_model.h5')
        elif Selected == 4:
            model = load_model('Models/cnn_model.h5')
        elif Selected == 5:
            model = load_model('Models/cnn_model.h5')
            
        w2v_model = Word2Vec.load('Models/word2vec.model')
        stemmer = ISRIStemmer()
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        with open("Stopwords/Stopwords_List.txt", "r", encoding="utf-8") as f:
            arabic_stopwords = set(f.read().splitlines())
            
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', ' ', text)
        tokens = tokenizer.tokenize(text)
        tokens = [token for token in tokens if token not in arabic_stopwords]
        tokens = [stemmer.stem(token) for token in tokens]
        tokens = [token.translate(str.maketrans("", "", string.punctuation)) for token in tokens]
        tokens = [token for token in tokens if not token.isdigit()]
        tokens = [token for token in tokens if token]
        return tokens
    
    def predict_category():
        #WARNING: MAINLY FOR CNN, USE YOU OWN CLASSIFICATION TECHNIQUE.
        if Selected == 1:
            print("Noice")
            article = Input_Textbox.get("1.0", "end")
            tokens = MainGUI.Preprocess_Text(article)
            sequence = [w2v_model.wv.key_to_index[token] if token in w2v_model.wv.key_to_index else 0 for token in tokens]
            sequence_padded = pad_sequences([sequence], maxlen=3843, padding="post")
            prediction = model.predict(sequence_padded)
            category = np.argmax(prediction)
            confidence = np.max(prediction)
            label = ['Politics', 'Entertainment', 'Economy', 'Sports'][category]
            result_label = customtkinter.CTkLabel(Main, text="", font=("System", 18, "bold"))
            result_label.configure(text=f"Predicted Category: {label}\nConfidence Score: {confidence:.2f}")
            result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
    
    def copy():
        Main.clipboard_clear()
        Main.clipboard_append(Input_Textbox.selection_get())

    def paste():
        Input_Textbox.insert(Main.INSERT, Main.clipboard_get())
    
    menu = Menu(customtkinter.CTk(), tearoff=0)
    menu.add_command(label="Copy", command=copy)
    menu.add_command(label="Paste", command=paste) 
          
    def DestroyAll():
        widgets = Main.winfo_children()
        for widget in widgets:
            if hasattr(widget, 'delete'):
                widget.delete(0, 'end')
            elif hasattr(widget, 'destroy'):
                widget.destroy()
                
    def Continue():
        MainGUI.DestroyAll()
        MainGUI.Train_Test_Form()
    
    def ResetWindow():
        Main.destroy()
        os.startfile(r"MainGUI.py")
    
    def ClearText():
        Input_Textbox.delete('1.0', 'end')

    def Go_Test():
        global classifier
        global Selected
        classifier = classifierComboBox.get()
        
        if classifier == "K-Nearst Neighbour":
            Selected = 1
            MainGUI.Create_Classifier_Form()
        elif classifier == "Support Vector Machine":
            Selected = 2
            MainGUI.Create_Classifier_Form()
        elif classifier == "Decision Tree":
            Selected = 3
            MainGUI.Create_Classifier_Form()
        elif classifier == "Convolutional Neural Networks":
            Selected = 4
            MainGUI.Create_Classifier_Form()
        elif classifier == "Recurrent Neural Network":
            Selected = 5
            MainGUI.Create_Classifier_Form()
        else:
            messagebox.showinfo("Error", "Please choose a classifier")
    
    def Go_Train():
        global Selected
        classifier = classifierComboBox.get()
        
        if classifier == "K-Nearst Neighbour":
            os.startfile(r"Classifiers\\KNN\\train.py")
            Selected = 1
        elif classifier == "Support Vector Machine":
            os.startfile(r"Classifiers\\SVM\\train.py")
            Selected = 2
        elif classifier == "Decision Tree":
            os.startfile(r"Classifiers\\DT\\train.py")
            Selected = 3
        elif classifier == "Convolutional Neural Networks":
            os.startfile(r"Classifiers\\CNN\\train.py")
            Selected = 4
        elif classifier == "Recurrent Neural Network":
            os.startfile(r"Classifiers\\RNN\\train.py")
            Selected = 5

    def TrainClassifierPage():
        MainGUI.DestroyAll()
        allClassifiers = ["...", "Convolutional Neural Networks", "K-Nearst Neighbour", "Support Vector Machine", "Recurrent Neural Network","Decision Tree"]
        ChooseClassLbl = customtkinter.CTkLabel(Main, text="Choose a Classifier for Training", font=("System", 40, "bold"))
        global classifierComboBox
        classifierComboBox = customtkinter.CTkComboBox(Main, values=allClassifiers, width=400, height=55, font=("System", 30, "bold"))
        ConfirmButton = customtkinter.CTkButton(Main, text="Confirm", command=lambda: MainGUI.Go_Train(), width=100, height=50, font=("System", 30, "bold"), fg_color="darkgreen")
        QuitButton = customtkinter.CTkButton(Main, text="Quit", command=quit, width=100, height=50, font=("System", 30, "bold"), fg_color="darkgreen")

        ChooseClassLbl.place(x=Main.winfo_screenwidth()/2 - 450,y=Main.winfo_screenheight()/2 - 400, anchor="center")
        classifierComboBox.place(x=Main.winfo_screenwidth()/2 - 520, y=Main.winfo_screenheight()/2 - 250, anchor="center")
        ConfirmButton.place(x=Main.winfo_screenwidth()/2 - 220, y=Main.winfo_screenheight() /2 - 250, anchor="center")
        QuitButton.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight() / 2 - 50, anchor="center")

    def TestClassifierPage():
        MainGUI.DestroyAll()
        allClassifiers = ["...", "Convolutional Neural Networks", "K-Nearst Neighbour", "Support Vector Machine", "Recurrent Neural Network",
                          "Decision Tree"]
        ChooseClassLbl = customtkinter.CTkLabel(Main, text="Choose a Classifier for Testing", font=("System", 40, "bold"))
        global classifierComboBox
        classifierComboBox = customtkinter.CTkComboBox(Main, values=allClassifiers, width=400, height=55, font=("System", 30, "bold"))
        ConfirmButton = customtkinter.CTkButton(Main, text="Confirm", command=lambda: MainGUI.Go_Test(), width=100, height=50, font=("System", 30, "bold"), fg_color="darkgreen")
        QuitButton = customtkinter.CTkButton(Main, text="Quit", command=quit, width=100, height=50, font=("System", 30, "bold"), fg_color="darkgreen")

        ChooseClassLbl.place(x=Main.winfo_screenwidth()/2 - 450,y=Main.winfo_screenheight()/2 - 400, anchor="center")
        classifierComboBox.place(x=Main.winfo_screenwidth()/2 - 520, y=Main.winfo_screenheight()/2 - 250, anchor="center")
        ConfirmButton.place(x=Main.winfo_screenwidth()/2 - 220, y=Main.winfo_screenheight() /2 - 250, anchor="center")
        QuitButton.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight() / 2 - 50, anchor="center")

    def Train_Test_Form():
        MainGUI.DestroyAll()
        ChooseButtonLabel = customtkinter.CTkLabel(Main, text="Would like to train or test?", font=("System", 40, "bold"))
        TrainButton = customtkinter.CTkButton(Main, text="Train", width=500, height=125, font=("System", 40, "bold"), fg_color="darkgreen", command=lambda: MainGUI.TrainClassifierPage())
        TestButton = customtkinter.CTkButton(Main, text="Test", width=500, height=125, font=("System", 40, "bold"), fg_color="darkgreen", command=lambda: MainGUI.TestClassifierPage())
        ChooseButtonLabel.place(x=Main.winfo_screenwidth()/2 - 450, y=Main.winfo_screenheight()/2 - 400, anchor="center")
        TrainButton.place(x=Main.winfo_screenwidth()/2 - 450,y=Main.winfo_screenheight() / 2 - 250, anchor="center")
        TestButton.place(x=Main.winfo_screenwidth()/2 -450 ,y=Main.winfo_screenheight() / 2 - 100, anchor="center")

    def Create_Classifier_Form():
        MainGUI.DestroyAll()
        ChooseFileLabel = customtkinter.CTkLabel(Main, text="Input An Article", font=("System", 40, "bold"))
        classifierLabel = customtkinter.CTkLabel(Main, text=classifier, font=("System", 10, "bold"))
        Classify_Button = customtkinter.CTkButton(Main, text="Classify",width=200, height=62, font=("System", 30, "bold"), fg_color="darkgreen", command=lambda:MainGUI.predict_category())
        Clear_Btn = customtkinter.CTkButton(Main, text="Clear",width=180, height=62, font=("System", 30, "bold"), fg_color="darkgreen", command=lambda:MainGUI.ClearText())

        global Input_Textbox
        Input_Textbox = customtkinter.CTkTextbox(Main, width=600, height=300, font=("System", 20, "bold"))
        # print("hello")        # PLACE
        ChooseFileLabel.place(x=Main.winfo_screenwidth()/2 - 450,y=Main.winfo_screenheight()/2 - 450, anchor="center")
        Input_Textbox.place(x=Main.winfo_screenwidth()/2 - 600,y=Main.winfo_screenheight()/2 - 200, anchor="center")
        Classify_Button.place(x=Main.winfo_screenwidth()/2 - 125,y=Main.winfo_screenheight()/2 - 300, anchor="center")
        classifierLabel.place(x=Main.winfo_screenwidth()/2 - 480, y=Main.winfo_screenheight() - 550, anchor="center")
        Clear_Btn.place(x=Main.winfo_screenwidth()/2 - 125, y=Main.winfo_screenheight() - 650, anchor="center")

Main = customtkinter.CTk()
Main.title("Arabic News Classifier")
Main.attributes("-topmost", True)

ScreenWidth = Main.winfo_screenwidth()
ScreenHeight = Main.winfo_screenheight()
Main.geometry("1000x580".format(ScreenWidth, ScreenHeight))

WelcomeLabel = customtkinter.CTkLabel(Main, text="Welcome to the\nMain Page", font=("System", 40, "bold"))
# SigntureLabel = customtkinter.CTkLabel(
#     Main, text="GUI Made by: Hubos", font=("System", 10, "bold"))
ContinueButton = customtkinter.CTkButton(Main, text="Continue", command=lambda: MainGUI.Continue(),  width=500, height=125, font=("System", 40, "bold"), fg_color="darkgreen")
QuitButton = customtkinter.CTkButton(Main, text="Quit", command=quit, width=500, height=125, font=("System", 40, "bold"), fg_color="darkgreen")
WelcomeLabel.place(x=ScreenWidth/2-450, y=ScreenHeight/2 - 450, anchor="center")
ContinueButton.place(x=ScreenWidth/2 - 450, y=ScreenHeight/2 - 250, anchor="center")
QuitButton.place(x=ScreenWidth/2 - 450, y=ScreenHeight/2 - 100, anchor="center")

Main.mainloop()
