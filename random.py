# Importing the libraries
from tkinter import *
from tkinter import messagebox
import os
import webbrowser

import numpy as np
import pandas as pd


class HyperlinkManager:
    def __init__(self, textwidget):
        self.textwidget = textwidget
        self.textwidget.tag_config("hyper", foreground="blue", underline=1)
        self.textwidget.tag_bind("hyper", "<Enter>", self._enter)
        self.textwidget.tag_bind("hyper", "<Leave>", self._leave)
        self.textwidget.tag_bind("hyper", "<Button-1>", self._click)
        self.reset()

    def reset(self):
        self.links = {}

    def add_hyperlink(self, start, end, url):
        self.links[self.textwidget.index(start)] = url
        self.textwidget.tag_add("hyper", start, end)

    def _enter(self, event):
        self.textwidget.config(cursor="hand2")

    def _leave(self, event):
        self.textwidget.config(cursor="")

    def _click(self, event):
        for index, url in self.links.items():
            if index <= self.textwidget.index(CURRENT):
                webbrowser.open(url)
                break


# Dataset and classifier initialization
df = pd.read_csv("Training.csv")
df.drop(["Unnamed: 133"], axis=1, inplace=True)
df.head()
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df["prognosis"]
np.ravel(y)
tr = pd.read_csv("Testing.csv")
tr.head()
testx = tr[cols]
testy = tr["prognosis"]
np.ravel(testy)

# Encoding labels
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
testy = encoder.fit_transform(testy)

# Implementing the classifier
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)

classifier.fit(x, y)

# Method to simulate the working of a Chatbot by extracting and formulating questions
def print_disease(node):
    # ... existing code for print_disease
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    #print(val)
    disease = labelencoder.inverse_transform(val[0])
    return disease


def recurse(node, depth):
    # ... existing code for recurse
    global val,ans
    global tree_,feature_name,symptoms_present
    indent = "  " * depth
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]
        yield name + " ?"
#       ans = input()
        ans = ans.lower()
        if ans == 'yes':
            val = 1
        if name not in symptoms_present:
            symptoms_present.append(name)
    else:
        val = 0
    # ... existing code for recurse
    strData=""
    present_disease = print_disease(tree_.value[node])
#   print( "You may have " +  present_disease )
#   print()
    strData="You may have :" +  str(present_disease)
    QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
    red_cols = dimensionality_reduction.columns 
    symptoms_given = red_cols[dimensionality_reduction.loc[present_disease].values[0].nonzero()]
#   print("symptoms present  " + str(list(symptoms_present)))
#   print()
    strData="symptoms present:  " + str(list(symptoms_present))
    QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#   print("symptoms given "  +  str(list(symptoms_given)) )  
#   print()
    strData="symptoms given: "  +  str(list(symptoms_given))
    QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
    confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
#   print("confidence level is " + str(confidence_level))
#   print()
    strData="confidence level is: " + str(confidence_level)
    QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#   print('The model suggests:')
#   print()
    strData='The model suggests:'
    QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
    row = doctors[doctors['disease'] == present_disease[0]]
#   print('Consult ', str(row['name'].values))
#   print()
    strData='Consult '+ str(row['name'].values)
    QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
#   print('Visit ', str(row['link'].values))
    #print(present_disease[0])
    hyperlink = HyperlinkManager(QuestionDigonosis.objRef.txtDigonosis)
    strData='Visit '+ str(row['link'].values[0])
    def click1():
        webbrowser.open_new(str(row['link'].values[0]))
    QuestionDigonosis.objRef.txtDigonosis.insert(INSERT, strData, hyperlink.add(click1))
    #QuestionDigonosis.objRef.txtDigonosis.insert(END,str(strData)+'\n')                  
    yield strData

    def tree_to_code(tree, feature_names):
        global tree_,feature_name,symptoms_present
        tree_ = tree.tree_
        #print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print("def tree({}):".format(", ".join(feature_names)))
        symptoms_present = []   
#        recurse(0, 1)

      
def execute_bot():
#print("Please reply with yes/Yes or no/No for the following symptoms")    
    tree_to_code(classifier,cols)


# This section of code to be run after scraping the data

doc_dataset = pd.read_csv('doctors_dataset.csv', names = ['Name', 'Description'])


diseases = dimensionality_reduction.index
diseases = pd.DataFrame(diseases)

doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

doctors['disease'] = diseases['prognosis']


doctors['name'] = doc_dataset['Name']
doctors['link'] = doc_dataset['Description']

record = doctors[doctors['disease'] == 'AIDS']
record['name']
record['link']




# Execute the bot and see it in Action
#execute_bot()


# ... existing code for doctors and diseases


class QuestionDiagnosis(Frame):
    objIter = None
    objRef = None

    def __init__(self, master=None):
        master.title("Question")
        # root.iconbitmap("")
        master.state("z")
        QuestionDiagnosis.objRef = self
        super().__init__(master=master)
        self["bg"] = "light blue"
        self.createWidget()
        self.iterObj = None

    def createWidget(self):
        self.txtQuestion = Text(
            self, width=80, height=10, wrap=WORD, background="white"
        )
        self.txtQuestion.grid(row=0, column=0, padx=10, pady=10)
        self.txtDiagnosis = Text(
            self, width=80, height=10, wrap=WORD, background="white"
        )
        self.txtDiagnosis.grid(row=1, column=0, padx=10, pady=10)
        self.btnYes = Button(self, text="Yes", command=self.btnYes_Click)
        self.btnYes.grid(row=2, column=0, padx=10, pady=10)
        self.btnNo = Button(self, text="No", command=self.btnNo_Click)
        self.btnNo.grid(row=2, column=1, padx=10, pady=10)
        self.btnStart = Button(self, text="Start Diagnosis", command=self.btnStart_Click)
        self.btnStart.grid(row=2, column=2, padx=10, pady=10)
        self.btnExit = Button(self, text="Exit", command=self.btnExit_Click)
        self.btnExit.grid(row=2, column=3, padx=10, pady=10)

    def btnYes_Click(self):
        global val, ans
        ans = 'yes'
        self.txtDiagnosis.delete(0.0, END)
        str1 = QuestionDiagnosis.objIter.__next__()
        self.txtQuestion.insert(END, str1 + "\n")

    def btnNo_Click(self):
        global val, ans
        ans = 'no'
        self.txtDiagnosis.delete(0.0, END)
        str1 = QuestionDiagnosis.objIter.__next__()
        self.txtQuestion.insert(END, str1 + "\n")

    def btnStart_Click(self):
        execute_bot()
        self.txtDiagnosis.delete(0.0, END)
        self.txtQuestion.delete(0.0, END)
        self.txtDiagnosis.insert(
            END, "Please Click on Yes or No for the Above symptoms in Question"
        )
        symptoms_present = []
        QuestionDiagnosis.objIter = recurse(0, 1, symptoms_present)
        str1 = QuestionDiagnosis.objIter.__next__()
        self.txtQuestion.insert(END, str1 + "\n")

    def btnExit_Click(self):
        root.destroy()


class MainForm(Frame):
    def __init__(self, master=None):
        master.title("Main Form")
        master.geometry("600x400")
        super().__init__(master=master)
        self.createWidget()

    def createWidget(self):
        self.lblHeading = Label(
            self,
            text="Medical Diagnosis System",
            font=("Arial", 20),
            pady=40,
            padx=20,
        )
        self.lblHeading.pack()
        self.btnDiagnose = Button(
            self,
            text="Start Diagnosis",
            font=("Arial", 14),
            pady=20,
            padx=20,
            command=self.btnDiagnose_Click,
        )
        self.btnDiagnose.pack()
        self.btnExit = Button(
            self,
            text="Exit",
            font=("Arial", 14),
            pady=20,
            padx=20,
            command=self.btnExit_Click,
        )
        self.btnExit.pack()

    def btnDiagnose_Click(self):
        QuestionDiagnosis(Toplevel())

    def btnExit_Click(self):
        root.destroy()


# Run the application
root = Tk()
root.geometry("800x600")
mainForm = MainForm(root)
mainForm.pack()
root.mainloop()
