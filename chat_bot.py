import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import re
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
from threading import Thread
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {}

# Load dataset and train models
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

model = SVC()
model.fit(x_train, y_train)

reduced_data = training.groupby(training['prognosis']).max()

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            try:
                severityDictionary[row[0]] = int(row[1])
            except (ValueError, IndexError):
                print(f"Skipping invalid row in severity CSV: {row}")

def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 1:
                description_list[row[0]] = row[1]

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 5:
                precautionDictionary[row[0]] = row[1:5]

def calc_condition(exp, days):
    total_severity = sum(severityDictionary.get(item, 0) for item in exp)
    if ((total_severity * int(days)) / (len(exp) + 1)) > 13:
        return "You should take consultation from a doctor."
    else:
        return "It might not be that bad but you should take precautions."

def check_pattern(dis_list, inp):
    inp = inp.replace(' ', '_')
    matches = [item for item in dis_list if re.search(inp, item)]
    return (len(matches) > 0), matches

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    model = DecisionTreeClassifier()
    model.fit(X, y)
    input_vector = np.zeros(len(X.columns))
    for symptom in symptoms_exp:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return model.predict([input_vector])[0]

# GUI setup
root = tk.Tk()
root.title("🤖 AI Healthcare Chatbot")
root.geometry("750x700")
root.config(bg="#1e1e1e")

style = ttk.Style()
style.configure("TButton", font=("Orbitron", 12), padding=10, relief="flat", background="#444444", foreground="#00e5ff")
style.map("TButton", background=[('active', '#00b8d4')])

header = tk.Label(root, text="AI-Based Healthcare Chatbot", font=("Orbitron", 24, "bold"), fg="#00e5ff", bg="#1e1e1e")
header.pack(pady=30)

form_frame = tk.Frame(root, bg="#1e1e1e")
form_frame.pack(pady=20)

tk.Label(form_frame, text="Enter Symptom:", font=("Orbitron", 14), fg="#00e5ff", bg="#1e1e1e").grid(row=0, column=0, sticky="e")
symptom_var = tk.StringVar()

# Symptom suggestions list
def update_symptom_suggestions(event):
    symptom_input = symptom_var.get().strip().lower()
    matched = [symptom for symptom in symptoms_dict.keys() if symptom_input in symptom.lower()]
    symptom_combobox['values'] = matched
    if len(matched) == 0:
        symptom_combobox.set('No matches found')

symptom_combobox = ttk.Combobox(form_frame, textvariable=symptom_var, width=40, font=("Orbitron", 12))
symptom_combobox.grid(row=0, column=1, pady=5)
symptom_combobox.bind('<KeyRelease>', update_symptom_suggestions)

tk.Label(form_frame, text="Number of Days:", font=("Orbitron", 14), fg="#00e5ff", bg="#1e1e1e").grid(row=1, column=0, sticky="e")
days_var = tk.StringVar()
tk.Entry(form_frame, textvariable=days_var, width=40, font=("Orbitron", 12)).grid(row=1, column=1, pady=5)

result_text = tk.Text(root, height=15, width=70, wrap='word', font=("Orbitron", 12), bg="#333333", fg="#00e5ff")
result_text.pack(pady=20)

progress = ttk.Progressbar(root, orient="horizontal", length=500, mode="indeterminate", style="TButton")
progress.pack(pady=10)

# Clear function to reset fields
def clear_fields():
    symptom_var.set("")
    days_var.set("")
    result_text.delete("1.0", tk.END)
    progress.stop()

def show_result_with_animation(text):
    result_text.delete("1.0", tk.END)
    delay = 0.05  # Delay in seconds between each character

    def update_text(index):
        if index < len(text):
            result_text.insert(tk.END, text[index])
            result_text.yview(tk.END)  # Scroll to the end of the text
            root.after(int(delay * 1000), update_text, index + 1)

    update_text(0)

def predict():
    symptom_input = symptom_var.get().strip().lower().replace(" ", "_")
    days_input = days_var.get().strip()

    if not symptom_input or not days_input:
        messagebox.showerror("Input Error", "Please fill all fields.")
        return

    try:
        days = int(days_input)
    except ValueError:
        messagebox.showerror("Input Error", "Days must be an integer.")
        return

    def run_prediction():
        progress.start(10)
        time.sleep(1)
        conf, matched = check_pattern(list(symptoms_dict.keys()), symptom_input)
        if not conf:
            progress.stop()
            messagebox.showerror("Symptom Error", "No matching symptoms found.")
            return

        selected_symptom = matched[0]
        symptoms_present = []
        tree_ = clf.tree_
        feature_name = [cols[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

        def recurse(node):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                val = 1 if name == selected_symptom else 0
                if val <= threshold:
                    recurse(tree_.children_left[node])
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node])
            else:
                present_disease = le.inverse_transform([np.argmax(tree_.value[node])])[0]
                symptoms_given = reduced_data.columns[reduced_data.loc[present_disease].values[0].nonzero()]
                symptoms_exp = [sym for sym in symptoms_given if messagebox.askyesno("Symptom Check", f"Do you have {sym.replace('_', ' ')}?")]

                secondary_disease = sec_predict(symptoms_exp)
                condition = calc_condition(symptoms_exp, days)

                result = f"\n🤖 You may have: {present_disease}\n"
                if present_disease != secondary_disease:
                    result += f"🤔 Or: {secondary_disease}\n"
                result += f"\n📖 Description:\n{description_list.get(present_disease, 'No description available.')}\n"
                if present_disease != secondary_disease:
                    result += f"{description_list.get(secondary_disease, '')}\n"

                precautions = precautionDictionary.get(present_disease, [])
                if precautions:
                    result += f"\n💡 Precautions:\n- " + "\n- ".join(precautions)
                result += f"\n\n📢 Advice: {condition}"

                # Show prediction confidence (accuracy)
                accuracy = model.score(x_test, y_test) * 100
                result += f"\n\n🔍 Accuracy: {accuracy:.2f}%"

                show_result_with_animation(result)
                progress.stop()

        recurse(0)

    Thread(target=run_prediction).start()

predict_button = ttk.Button(root, text="🤖 Predict Disease", command=predict)
predict_button.pack(pady=10)

clear_button = ttk.Button(root, text="❌ Clear", command=clear_fields)
clear_button.pack(pady=10)

getSeverityDict()
getDescription()
getprecautionDict()

root.mainloop()
