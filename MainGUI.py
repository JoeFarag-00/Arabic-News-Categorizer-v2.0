import customtkinter
import os
from tkinter import messagebox
from tkinter import filedialog
import pickle
customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('green')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import string
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from tkinter import Tk, Label, Text, Button, Menu
import random

class MainGUI():
    
    @staticmethod
    def Preprocess_Text(text):
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
        ## WARNING: MAINLY FOR CNN, USE YOU OWN CLASSIFICATION TECHNIQUE.
        if Selected == 1:
            model = load_model('Models/CNN/cnn_model.h5')
            embedding = Word2Vec.load('Models/CNN/word2vec.model')
            article = Input_Textbox.get("1.0", "end")
            tokens = MainGUI.Preprocess_Text(article)
            sequence = [embedding.wv.key_to_index[token] if token in embedding.wv.key_to_index else 0 for token in tokens]
            sequence_padded = pad_sequences([sequence], maxlen=3843, padding="post")
            prediction = model.predict(sequence_padded)
            category = np.argmax(prediction)
            confidence = np.max(prediction)
            label = ['Politics', 'Entertainment', 'Economy', 'Sports'][category]
            result_label = customtkinter.CTkLabel(Main, text="", font=("System", 18, "bold"))
            result_label.configure(text=f"")
            result_label.configure(text=f"Predicted Category: {label}\nConfidence Score: {confidence:.2f}")
            result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
            
        if Selected == 2: #BROKEN SVM TEST
            tfidf_vectorizer = joblib.load("Models/SVM/tfidf.pkl")
            svm_model = joblib.load("Models/SVM/svm_model.pkl")
            input_text = Input_Textbox.get("1.0", "end")
            preprocessed_text = MainGUI.Preprocess_Text(input_text)
            vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
            predicted_category = svm_model.predict(vectorized_text)[0]
            result_label.configure(text=f"")
            result_label.configure(text=f"Predicted Category: {predicted_category}")
            result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
                    
        if Selected == 3:
            dt_vec = joblib.load("Models/DT/DT_tfidf_vectorizer.pkl")
            dt_model = joblib.load("Models/DT/DT_model.pkl")
            def predict_DT():
                input_text = Input_Textbox.get("1.0", "end")
                if input_text:
                    preprocessed_text = MainGUI.Preprocess_Text(input_text)
                    preprocessed_text = ' '.join(preprocessed_text)
                    vectorized_text = dt_vec.transform([preprocessed_text])
                    predicted_prob = dt_model.predict_proba(vectorized_text)[0]
                    predicted_category = dt_model.predict(vectorized_text)[0]
                    result_label = customtkinter.CTkLabel(Main, text="", font=("System", 18, "bold"))
                    result_label.configure(text=f"")
                    result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
                    result_label.configure(text=f"Predicted Category: {predicted_category}\nConfidence Score: {predicted_prob.max():.2%}")
                else:
                    result_label.configure(text="Please enter some text.")
                    result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
            predict_DT()
            
            
        if Selected == 4:
            lstm_w2v_model = Word2Vec.load('Models/LSTM/word2vec.model')
            lstm_model = load_model('Models/LSTM/lstm_model.h5')
            def predict_LSTM():
                article = Input_Textbox.get("1.0", "end")
                tokens = MainGUI.Preprocess_Text(article)
                sequence = [lstm_w2v_model.wv.key_to_index[token] if token in lstm_w2v_model.wv.key_to_index else 0 for token in tokens]
                sequence_padded = pad_sequences([sequence], maxlen=3843, padding="post")
                prediction = lstm_model.predict(sequence_padded)
                category = np.argmax(prediction)
                confidence = np.max(prediction)
                label = ['Politics', 'Entertainment', 'Economy', 'Sports'][category]
                result_label = customtkinter.CTkLabel(Main, text="", font=("System", 18, "bold"))
                result_label.configure(text=f"")
                result_label.configure(text=f"Predicted Category: {label}\nConfidence Score: {confidence:.2f}")
                result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
            predict_LSTM()
        
        if Selected == 5:
            tfidf = joblib.load("Models/KNN/tfidf.pkl")
            knn_model = joblib.load("Models/KNN/KNN_model.pkl")
            def predict_KNN():
                article = Input_Textbox.get("1.0", "end")
                article = tfidf.transform([article])
                prediction = knn_model.predict(article)[0]
                confidence = knn_model.predict_proba(article).max() * 100
                result_label = customtkinter.CTkLabel(Main, text="", font=("System", 18, "bold"))
                result_label.configure(text=f"")
                result_label.configure(text=f"Predicted category: {prediction}\nConfidence: {confidence:.2f}%")
                result_label.place(x=Main.winfo_screenwidth()/2 - 120,y=Main.winfo_screenheight()/2 - 200, anchor="center")
            predict_KNN()
        
        if Selected == 6:
           
           pass
        
                
    def copy():
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
        
        if classifier == "Convolutional Neural Networks":
            Selected = 1
            print("CNN")
            MainGUI.Create_Classifier_Form()
            
        elif classifier == "Support Vector Machine":
            Selected = 2
            print("SVM")
            MainGUI.Create_Classifier_Form()
            
        elif classifier == "Decision Tree":
            Selected = 3
            MainGUI.Create_Classifier_Form()
            
        elif classifier == "Recurrent Neural Network":
            Selected = 4
            MainGUI.Create_Classifier_Form()
            
        elif classifier == "K-Nearst Neighbour":
            Selected = 5
            MainGUI.Create_Classifier_Form()
            
        else:
            messagebox.showinfo("Error", "Please choose a classifier")
    
    def Go_Train():
        global Selected
        classifier = classifierComboBox.get()
        
        if classifier == "Convolutional Neural Networks":
            os.startfile(r"Classifiers\\CNN\\train.py")
            Selected = 1
        elif classifier == "Support Vector Machine":
            os.startfile(r"Classifiers\\SVM\\train.py")
            Selected = 2
        elif classifier == "Decision Tree":
            os.startfile(r"Classifiers\\DT\\train.py")
            Selected = 3
        elif classifier == "Recurrent Neural Network":
            os.startfile(r"Classifiers\\RNN\\train.py")
            Selected = 4
        elif classifier == "K-Nearst Neighbour":
            os.startfile(r"Classifiers\\KNN\\train.py")
            Selected = 5

    def TrainClassifierPage():
        MainGUI.DestroyAll()
        allClassifiers = ["...", "Convolutional Neural Networks", "Support Vector Machine","Decision Tree", "Recurrent Neural Network","K-Nearst Neighbour"]
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
        allClassifiers = ["...", "Convolutional Neural Networks", "Support Vector Machine","Decision Tree", "Recurrent Neural Network","K-Nearst Neighbour"]
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
ContinueButton = customtkinter.CTkButton(Main, text="Continue", command=lambda: MainGUI.Continue(),  width=500, height=125, font=("System", 40, "bold"), fg_color="darkgreen")
QuitButton = customtkinter.CTkButton(Main, text="Quit", command=quit, width=500, height=125, font=("System", 40, "bold"), fg_color="darkgreen")
WelcomeLabel.place(x=ScreenWidth/2-450, y=ScreenHeight/2 - 450, anchor="center")
ContinueButton.place(x=ScreenWidth/2 - 450, y=ScreenHeight/2 - 250, anchor="center")
QuitButton.place(x=ScreenWidth/2 - 450, y=ScreenHeight/2 - 100, anchor="center")

Main.mainloop()
