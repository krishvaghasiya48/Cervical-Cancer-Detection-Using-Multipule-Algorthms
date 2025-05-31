import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

class CervicalCancerDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cervical Cancer Detection System")
        self.root.geometry("800x600")

        # Initialize model and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # Binary classification: Cancer/No Cancer
        self.model = self.model.to(self.device)

        # Load the model checkpoint
        checkpoint_path = 'cervical_cancer_detection_binary_resnet50.pth'
        if os.path.exists(checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {str(e)}")
        else:
            print(f"Checkpoint file {checkpoint_path} not found. Please train the model first.")

        # Define class names
        self.classes = ['No Cancer', 'Cancer']

        # Image transformation for the model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create GUI elements
        self.create_gui()

    def create_gui(self):
        # Main frame with scrolling
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind_all("<MouseWheel>", lambda event: self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfig(self.canvas_frame, width=event.width))

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Header section
        self.header_frame = ttk.Frame(self.scrollable_frame)
        self.header_frame.pack(fill=tk.X, padx=10, pady=10)

        self.title_label = ttk.Label(self.header_frame, text="Cervical Cancer Detection System", font=('Helvetica', 16, 'bold'))
        self.title_label.pack(pady=5)

        self.status_label = ttk.Label(
            self.header_frame,
            text="Ready to analyze images" if os.path.exists('cervical_cancer_detection_binary_resnet50.pth') else "Please train the model first",
            font=('Helvetica', 10)
        )
        self.status_label.pack(pady=5)

        self.upload_button = ttk.Button(self.header_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=5)

        # Image display section
        self.image_frame = ttk.LabelFrame(self.scrollable_frame, text="Uploaded Image", padding="10")
        self.image_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(pady=10)

        # Results section
        self.results_frame = ttk.LabelFrame(self.scrollable_frame, text="Detection Results", padding="10")
        self.results_frame.pack(fill=tk.X, padx=10, pady=10)

        self.progress_vars = {}
        self.progress_labels = {}

        for class_name in self.classes:
            frame = ttk.Frame(self.results_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=class_name, width=25)
            label.pack(side=tk.LEFT)

            self.progress_vars[class_name] = tk.DoubleVar()
            progress = ttk.Progressbar(frame, variable=self.progress_vars[class_name], maximum=100, length=300)
            progress.pack(side=tk.LEFT, padx=5)

            percentage = ttk.Label(frame, text="0%", width=10)
            self.progress_labels[class_name] = percentage
            percentage.pack(side=tk.LEFT)

        # Conclusion section
        self.conclusion_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Conclusion", padding="10")
        self.conclusion_frame.pack(fill=tk.X, padx=10, pady=10)

        self.desc_label = ttk.Label(
            self.conclusion_frame,
            text=(
                "Detection Results:\n"
                "- The green bars above show the model's confidence for each class (inverted for demonstration).\n"
                "- 'Cell Type' displays the most likely result based on inverted probabilities.\n"
                "- 'Cancer Risk Assessment' and the message below explain the risk and recommended action.\n"
                "If the result is 'Uncertain', please consult a healthcare provider."
            ),
            font=('Helvetica', 9),
            foreground="gray"
        )
        self.desc_label.pack(pady=(0, 5))

        self.cell_type_label = ttk.Label(self.conclusion_frame, text="Cell Type: Not analyzed yet", font=('Helvetica', 12))
        self.cell_type_label.pack(pady=5)

        self.risk_label = ttk.Label(self.conclusion_frame, text="Cancer Risk Assessment: Not analyzed yet", font=('Helvetica', 12))
        self.risk_label.pack(pady=5)

        self.conclusion_text = tk.Text(self.conclusion_frame, height=4, width=50, wrap=tk.WORD, font=('Helvetica', 10))
        self.conclusion_text.pack(pady=5)
        self.conclusion_text.insert(tk.END, "Please upload an image for analysis.")
        self.conclusion_text.config(state=tk.DISABLED)

    def upload_image(self):
        if not os.path.exists('cervical_cancer_detection_binary_resnet50.pth'):
            messagebox.showwarning("Model Not Found", "Please train the model first or ensure the model file is in the project directory.")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tif *.tiff"),
                ("GIF files", "*.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                image = Image.open(file_path).convert('RGB')
                self.display_image(image)
                self.status_label.config(text="Analyzing image...")
                self.root.update()
                self.predict(image)
                self.status_label.config(text="Analysis complete")
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                self.status_label.config(text="Error during analysis")

    def display_image(self, image):
        display_size = (300, 300)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference

    def __init__(self, root):
        self.root = root
        self.root.title("Cervical Cancer Detection System")
        self.root.geometry("800x600")

        # Initialize model and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # Binary classification: Cancer/No Cancer
        self.model = self.model.to(self.device)

        # Load the model checkpoint
        checkpoint_path = 'cervical_cancer_detection_binary_resnet50.pth'
        if os.path.exists(checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {str(e)}")
        else:
            print(f"Checkpoint file {checkpoint_path} not found. Please train the model first.")

        # Define class names
        self.classes = ['No Cancer', 'Cancer']

        # Image transformation for the model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create GUI elements
        self.create_gui()

        # Initialize prediction tracking
        self.total_images = 0
        self.correct_predictions = 0

    def create_gui(self):
        # Main frame with scrolling
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind_all("<MouseWheel>", lambda event: self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfig(self.canvas_frame, width=event.width))

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Header section
        self.header_frame = ttk.Frame(self.scrollable_frame)
        self.header_frame.pack(fill=tk.X, padx=10, pady=10)

        self.title_label = ttk.Label(self.header_frame, text="Cervical Cancer Detection System", font=('Helvetica', 16, 'bold'))
        self.title_label.pack(pady=5)

        self.status_label = ttk.Label(
            self.header_frame,
            text="Ready to analyze images" if os.path.exists('cervical_cancer_detection_binary_resnet50.pth') else "Please train the model first",
            font=('Helvetica', 10)
        )
        self.status_label.pack(pady=5)

        self.upload_button = ttk.Button(self.header_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=5)

        # Image display section
        self.image_frame = ttk.LabelFrame(self.scrollable_frame, text="Uploaded Image", padding="10")
        self.image_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(pady=10)

        # Results section
        self.results_frame = ttk.LabelFrame(self.scrollable_frame, text="Detection Results", padding="10")
        self.results_frame.pack(fill=tk.X, padx=10, pady=10)

        self.progress_vars = {}
        self.progress_labels = {}

        for class_name in self.classes:
            frame = ttk.Frame(self.results_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=class_name, width=25)
            label.pack(side=tk.LEFT)

            self.progress_vars[class_name] = tk.DoubleVar()
            progress = ttk.Progressbar(frame, variable=self.progress_vars[class_name], maximum=100, length=300)
            progress.pack(side=tk.LEFT, padx=5)

            percentage = ttk.Label(frame, text="0%", width=10)
            self.progress_labels[class_name] = percentage
            percentage.pack(side=tk.LEFT)

        # Conclusion section
        self.conclusion_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Conclusion", padding="10")
        self.conclusion_frame.pack(fill=tk.X, padx=10, pady=10)

        self.desc_label = ttk.Label(
            self.conclusion_frame,
            text=(
                "Detection Results:\n"
                "- The green bars above show the model's confidence for each class (inverted for demonstration).\n"
                "- 'Cell Type' displays the most likely result based on inverted probabilities.\n"
                "- 'Cancer Risk Assessment' and the message below explain the risk and recommended action.\n"
                "If the result is 'Uncertain', please consult a healthcare provider."
            ),
            font=('Helvetica', 9),
            foreground="gray"
        )
        self.desc_label.pack(pady=(0, 5))

        self.cell_type_label = ttk.Label(self.conclusion_frame, text="Cell Type: Not analyzed yet", font=('Helvetica', 12))
        self.cell_type_label.pack(pady=5)

        self.risk_label = ttk.Label(self.conclusion_frame, text="Cancer Risk Assessment: Not analyzed yet", font=('Helvetica', 12))
        self.risk_label.pack(pady=5)

        self.conclusion_text = tk.Text(self.conclusion_frame, height=4, width=50, wrap=tk.WORD, font=('Helvetica', 10))
        self.conclusion_text.pack(pady=5)
        self.conclusion_text.insert(tk.END, "Please upload an image for analysis.")
        self.conclusion_text.config(state=tk.DISABLED)

    def upload_image(self):
        if not os.path.exists('cervical_cancer_detection_binary_resnet50.pth'):
            messagebox.showwarning("Model Not Found", "Please train the model first or ensure the model file is in the project directory.")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tif *.tiff"),
                ("GIF files", "*.gif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                image = Image.open(file_path).convert('RGB')
                self.display_image(image)
                self.status_label.config(text="Analyzing image...")
                self.root.update()
                self.predict(image, file_path)
                self.status_label.config(text="Analysis complete")
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                self.status_label.config(text="Error during analysis")

    def predict(self, image, file_path=None):
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Get model prediction
            with torch.no_grad():
                self.model.eval()
                output = self.model(input_tensor)
                prob_cancer = torch.sigmoid(output).item()
                prob_no_cancer = 1 - prob_cancer

            # Invert the probabilities
            inverted_prob_cancer = prob_no_cancer  # Originally "No Cancer" probability becomes "Cancer"
            inverted_prob_no_cancer = prob_cancer  # Originally "Cancer" probability becomes "No Cancer"

            # Update progress bars with inverted probabilities
            self.progress_vars['No Cancer'].set(inverted_prob_no_cancer * 100)
            self.progress_labels['No Cancer'].config(text=f"{inverted_prob_no_cancer * 100:.1f}%")
            self.progress_vars['Cancer'].set(inverted_prob_cancer * 100)
            self.progress_labels['Cancer'].config(text=f"{inverted_prob_cancer * 100:.1f}%")

            # Determine predicted class based on inverted probabilities
            predicted_class = 'Cancer' if inverted_prob_cancer >= inverted_prob_no_cancer else 'No Cancer'
            probability = max(inverted_prob_cancer, inverted_prob_no_cancer)

            # Update conclusion with the inverted prediction
            self.update_conclusion(predicted_class, probability)

            # Update accuracy tracking
            self.total_images += 1

            # Determine true label from filename or folder name (assuming filename contains label)
            true_label = None
            if file_path:
                lower_path = file_path.lower()
                if 'cancer' in lower_path:
                    true_label = 'Cancer'
                elif 'no_cancer' in lower_path or 'nocancer' in lower_path or 'normal' in lower_path:
                    true_label = 'No Cancer'

            if true_label is not None:
                if true_label == predicted_class:
                    self.correct_predictions += 1

                accuracy = self.correct_predictions / self.total_images * 100
                accuracy_message = f"Accuracy: {accuracy:.2f}%"
                print(accuracy_message)
                self.status_label.config(text=accuracy_message)
                self.root.update_idletasks()
            else:
                unknown_message = "Accuracy: N/A"
                print(unknown_message)
                self.status_label.config(text=unknown_message)
                self.root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")
            self.status_label.config(text="Error during analysis")

    def update_conclusion(self, predicted_class, probability):
        confidence_threshold = 60.0
        predicted_prob = probability * 100

        # Remove cell type label update to not print cell type in GUI
        # self.cell_type_label.config(text=f"Cell Type: {predicted_class}", foreground="black")

        # Determine risk level and conclusion based on the inverted prediction
        if predicted_prob < confidence_threshold:
            risk_text = "Uncertain"
            risk_color = "orange"
            conclusion = (
                f"UNCERTAIN detected with low confidence ({predicted_prob:.1f}%). "
                "The analysis is inconclusive. Please consult a healthcare provider for proper evaluation."
            )
        else:
            if predicted_class == 'Cancer':
                risk_text = "High Risk"
                risk_color = "red"
                conclusion = (
                    f"WARNING detected with {predicted_prob:.1f}% confidence. "
                    "This indicates a high risk of cervical cancer. Immediate medical consultation is recommended."
                )
            else:
                risk_text = "Low Risk"
                risk_color = "green"
                conclusion = (
                    f"NORMAL detected with {predicted_prob:.1f}% confidence. "
                    "This indicates normal cervical cells. Continue regular check-ups as recommended by your healthcare provider."
                )

        # Update risk label and conclusion text
        self.risk_label.config(text=f"Cancer Risk Assessment: {risk_text}", foreground=risk_color)
        self.conclusion_text.config(state=tk.NORMAL)
        self.conclusion_text.delete(1.0, tk.END)
        self.conclusion_text.insert(tk.END, conclusion)
        self.conclusion_text.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = CervicalCancerDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()