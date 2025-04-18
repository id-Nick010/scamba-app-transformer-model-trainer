import xml.etree.ElementTree as ET
import tkinter as tk
import csv

tree = ET.parse('dataset/data/sms-20250415202031.xml')

root = tree.getroot()

sms_list = root.findall('sms')

sms_bodies = []
for sms in sms_list:
    sms_bodies.append(sms.get('body'))
print("total imported dataset: " + str(len(sms_bodies)))

uni_sms_bodies = set(sms_bodies)
print("total unique sms: " + str(len(uni_sms_bodies)))
uni_sms_bodies = list(uni_sms_bodies)


texts = uni_sms_bodies
index = 0  # Track the current text position

scam_dataset = []
ham_dataset = []

def next_text(kind):
    global index
    if index < len(texts) - 1:
        if(kind == "scam"):
            scam_dataset.append(texts[index])
            print("String Scam")
        elif(kind == "ham"):
            ham_dataset.append(texts[index])
            print("String Ham")
        elif(kind == "deny"):
            print("String Denied")
        index += 1
        label.config(text=texts[index])
    else:
        label.config(text="No more messages!")  # End message

def key_pressed(event):
    # Trigger the next_text function when keys 1, 2, or 3 are pressed.
    if event.char == "1":
        next_text("scam")
    elif event.char == "2":
        next_text("deny")
    elif event.char == "3":
        next_text("ham")

# Create the main window
root = tk.Tk()
root.title("SMS Traversal UI Mechanics")

# Create a label to display text
label = tk.Label(root, text=texts[index], font=("Arial", 16))
label.pack(pady=20)

root.bind("<Key>", key_pressed)

# Create three buttons that trigger the same function
btn1 = tk.Button(root, text="Scam", command=lambda: next_text("scam"))
btn2 = tk.Button(root, text="Deny", command=lambda: next_text("deny"))
btn3 = tk.Button(root, text="Ham", command=lambda: next_text("ham"))

btn1.pack(side="left", padx=20)
btn2.pack(side="left", padx=20)
btn3.pack(side="left", padx=20)

# Run the UI loop
root.mainloop()

print("# of Scam: " + str(len(scam_dataset)) )
print("# of Ham: " + str(len(ham_dataset)) )

with open("scam_output.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    for item in scam_dataset:
        writer.writerow([item])  # Appends the list as a row
    print("Scam Data Written...")

with open("ham_output.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    for item in ham_dataset:
        writer.writerow([item])  # Appends the list as a row
    print("Ham Data Written...")