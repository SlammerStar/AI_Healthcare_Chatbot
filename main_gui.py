import tkinter as tk
from tkinter import messagebox
import threading
from your_disease_prediction_module import tree_to_code, getInfo, getSeverityDict, getDescription, getprecautionDict, clf, cols

class HealthCareChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HealthCare Chatbot")
        self.root.geometry("700x500")
        self.root.configure(bg="#f0f0f0")

        self.title = tk.Label(root, text="🩺 AI HealthCare Chatbot", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#2e7d32")
        self.title.pack(pady=20)

        self.instruction = tk.Label(root, text="Describe your symptom below:", font=("Helvetica", 12), bg="#f0f0f0")
        self.instruction.pack()

        self.input_text = tk.Entry(root, width=60, font=("Helvetica", 14))
        self.input_text.pack(pady=10)

        self.submit_button = tk.Button(root, text="Predict Disease", command=self.run_chatbot_logic, bg="#2e7d32", fg="white", font=("Helvetica", 12))
        self.submit_button.pack(pady=10)

        self.output_box = tk.Text(root, height=15, width=80, font=("Helvetica", 11))
        self.output_box.pack(pady=10)

    def run_chatbot_logic(self):
        symptom_input = self.input_text.get()
        if not symptom_input:
            messagebox.showwarning("Input Error", "Please enter your symptom.")
            return

        self.output_box.delete("1.0", tk.END)
        self.output_box.insert(tk.END, "Processing your input...\n")

        threading.Thread(target=self.process_prediction, args=(symptom_input,), daemon=True).start()

    def process_prediction(self, symptom_input):
        try:
            import sys
            from io import StringIO

            # Capture the output
            old_stdout = sys.stdout
            result = sys.stdout = StringIO()

            tree_to_code(clf, cols)  # Here you'd ideally customize to pass `symptom_input`

            sys.stdout = old_stdout
            output = result.getvalue()

            self.output_box.delete("1.0", tk.END)
            self.output_box.insert(tk.END, output)

        except Exception as e:
            self.output_box.insert(tk.END, f"Error: {str(e)}")

if __name__ == "__main__":
    getSeverityDict()
    getDescription()
    getprecautionDict()
    getInfo()  # Optional: remove or customize for GUI input

    root = tk.Tk()
    app = HealthCareChatbotApp(root)
    root.mainloop()
