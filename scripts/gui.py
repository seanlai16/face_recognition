import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import preprocess
import feature_extraction
import classification

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        
        # Buttons
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=10, pady=10)
        
        self.preprocess_button = tk.Button(root, text="Pre-process", command=self.preprocess_image)
        self.preprocess_button.grid(row=0, column=1, padx=10, pady=10)
        
        self.extract_button = tk.Button(root, text="Extract Feature", command=self.extract_feature)
        self.extract_button.grid(row=0, column=2, padx=10, pady=10)
        
        self.classify_button = tk.Button(root, text="Classification & Result", command=self.classify_image)
        self.classify_button.grid(row=0, column=3, padx=10, pady=10)
        
        # Placeholders for images
        self.placeholders = []
        for i in range(4):
            image = Image.open('../assets/placeholder.jpg')
            image = image.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            if i == 3:
                placeholder = Label(root, borderwidth=2, height=10, width=20, relief="groove", text="-")
                placeholder.grid(row=1, column=i, padx=10, pady=10)
                self.placeholders.append(placeholder)
            else:
                placeholder = Label(root, borderwidth=2, relief="groove")
                placeholder.config(image=photo)
                placeholder.image = photo
                placeholder.grid(row=1, column=i, padx=10, pady=10)
                self.placeholders.append(placeholder)
        
        self.image_paths = [""] * 4  # To store image paths

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.display_image(file_path, 0)  # Load image in the first placeholder
            self.image_paths[0] = file_path

    def preprocess_image(self):
        if self.image_paths[0]:
            preprocessed_path = '../assets/preprocessed.jpg'
            preprocess.preprocess_image(self.image_paths[0], preprocessed_path, 200)
            self.image_paths[1] = preprocessed_path
            self.display_image(self.image_paths[1], 1)  # Display preprocessed image in the second placeholder

    def extract_feature(self):
        if self.image_paths[1]:
            extracted_path = '../assets/extracted.jpg'
            feature_extraction.plot_landmarks(self.image_paths[1], extracted_path)
            self.image_paths[2] = extracted_path
            self.display_image(self.image_paths[2], 2)  # Display extracted features in the third placeholder

    def classify_image(self):
        if self.image_paths[2]:
            result, confidence = classification.classify(self.image_paths[1])
            formatted_result = '{result}\n{confidence}'.format(result=result, confidence=confidence)
            self.placeholders[3].config(text=formatted_result)
            self.placeholders[3].text = formatted_result

    def display_image(self, path, index):
        image = Image.open(path)
        image = image.resize((200, 200), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.placeholders[index].config(image=photo)
        self.placeholders[index].image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
