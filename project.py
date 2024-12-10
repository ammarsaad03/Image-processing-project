import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing GUI")
        self.center_window(700, 500)
        self.image_path = tk.StringVar(value="No file uploaded")
        self.original_img = None
        self.grey_img = None
        self.modified_img = None
        self.original_histogram=None
        self.create_widgets()
        self.initialize_operations()
        
    def center_window(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        group_frame = ttk.LabelFrame(self.root, text="Controls", padding=(10, 10))
        group_frame.pack(side="top", padx=10, pady=10, fill="y")

        image_path_label = ttk.Label(group_frame, textvariable=self.image_path, width=60, anchor="w", padding=(5, 5), relief="solid", borderwidth=2)
        image_path_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        upload_button = ttk.Button(group_frame, text="Upload Image", width=20, command=self.upload_image)
        upload_button.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(group_frame, text="Operation:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.operation_var = tk.StringVar()
        self.operation_combobox = ttk.Combobox(group_frame, textvariable=self.operation_var, state="readonly", width=30)
        self.operation_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.operation_combobox.bind("<<ComboboxSelected>>", self.update_algorithms)

        ttk.Label(group_frame, text="Algorithm:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.algorithm_var = tk.StringVar()
        self.algorithm_combobox = ttk.Combobox(group_frame, textvariable=self.algorithm_var, state="readonly", width=30)
        self.algorithm_combobox.grid(row=2, column=1, padx=5, pady=5)

        apply_button = ttk.Button(group_frame, text="Apply", width=20, command=self.apply_algorithm)
        apply_button.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        self.results_frame = ttk.LabelFrame(self.root, text="Results", padding=(10, 10))
        self.results_frame.pack(side="top", padx=10, pady=10, fill="both", expand=True)
        
        # Original Image Frame (Initially hidden)
        self.original_frame = tk.LabelFrame(self.results_frame, text="Original Image", padx=10, pady=10)
        self.original_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nwe")
        self.original_frame.grid_remove()  # Hide initially
        
        # Modified Image Frame (Initially hidden)
        self.modified_frame = tk.LabelFrame(self.results_frame, text="Modification of Image", padx=10, pady=10)
        self.modified_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nwe")
        self.modified_frame.grid_remove()  # Hide initially
        self.modified_panel=None
        
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(1, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=1)

    def initialize_operations(self):
        self.operations = {
            "Threshold": ["Threshold"],
            "Halftone": ["Simple Halftone", "Advanced Halftone (Error Diffusion)"],
            "Histogram": ["Histogram", "Histogram Equalization"],
            "Simple Edge Detection Methods": ["Sobel Operator", "Prewitt Operator", "Kirsch Compass Masks"],
            "Advanced Edge Detection Methods": ["Homogeneity Operator", "Difference Operator","Difference of Gaussians","Contrast Based","Variance","Range"],
            "Filtering": ["High-Pass Filter", "Low-Pass Filter", "Median Filter"],
            "Image Operations": ["Invert the Image", "Add Image", "Subtract Image"],
            "Histogram Based Segmentaion" : ["Manual Technique","Histogram peak Technique","Histogram Valley Technique" ,"Adaptive Histogram Technique"]
        }
        self.operation_combobox["values"] = list(self.operations.keys())
    def clear_panel(self, frame):
        # Destroy any existing widget (image label) inside the frame
        for widget in frame.winfo_children():
            widget.destroy()
    def hide_original_frame(self):
        self.original_frame.pack_forget()
    def hide_modified_frame(self):
        self.modified_frame.pack_forget()
        
    def reset_panels(self):
        self.hide_original_frame()
        self.hide_modified_frame()
        self.original_panel.destroy()
        if self.modified_panel:
            self.modified_panel.destroy()

    def upload_image(self):
        #uploading an image
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")])
        self.clear_panel(self.original_frame)
        self.hide_original_frame()
        if file_path:
            self.image_path.set(file_path)
            self.original_img = cv2.imread(file_path)
            self.original_img = cv2.resize(self.original_img, (300, 300))
            img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.clear_panel(self.original_frame)
            self.original_frame.pack(side="left",fill=tk.X,expand=1)
            self.original_panel = tk.Label(self.original_frame, image=img)
            self.original_panel.image = img  # Keep a reference to avoid garbage collection
            self.original_panel.pack(fill=tk.BOTH, expand=True)
            self.hide_modified_frame()
            self.convert_to_grayscale()
            self.original_histogram= cv2.calcHist([self.grey_img], [0], None, [256], [0, 256]).flatten().astype(int)
            
        else:
            self.image_path.set("No file uploaded")
            self.reset_panels()
            self.algorithm_var.set('')
            self.operation_var.set('')
            self.algorithm_combobox["values"] =""
        
    def display(self,custom_hist,hist_type):
        
        if self.modified_img is not None:
            # Create a new figure with two subplots (one for each histogram)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns
            
            rangey=np.arange(256)
            
            # custom histogram plot
            ax1.set_title("Custom Histogram")
            ax1.set_xlabel("Pixel Intensity")
            ax1.set_ylabel("Frequency")
            ax1.bar(rangey, custom_hist, width=1, color='black')
            ax1.set_xlim([0, 256])
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
            # Built-in histogram plot
            ax2.set_title("original Histogram")
            ax2.set_xlabel("Pixel Intensity")
            ax2.set_ylabel("Frequency")
            ax2.bar(rangey, self.original_histogram, width=1, color='red')
            ax2.set_xlim([0, 256])
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
            # Create a new window for the histograms
            hist_window = tk.Toplevel(self.root)
            text= hist_type +" Comparison"
            hist_window.title(text)
            hist_window.geometry("800x500")
    
            # Embed the matplotlib figure in the Tkinter window using FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, hist_window)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_algorithms(self, event=None):
        selected_operation = self.operation_var.get()
        algorithms = self.operations.get(selected_operation, [])
        self.algorithm_combobox["values"] = algorithms
        self.algorithm_var.set('')

    def apply_algorithm(self):
        
        if self.image_path.get() == "No file uploaded":
            messagebox.showwarning("Warning", "Please upload an image before applying an algorithm.")
            return
        self.clear_panel(self.modified_frame)
        self.convert_to_grayscale()
        selected_algorithm = self.algorithm_var.get()
        if not selected_algorithm:
            self.clear_panel(self.modified_frame)
            self.hide_modified_frame()
            messagebox.showwarning("Warning", "Please choose an algorithm.")
            return
        
        if selected_algorithm == "Histogram":
            self.modified_img=self.grey_img
            custom_hist = self.custom_histogram()
            self.display(custom_hist, "Histogram")
            pass
        self.clear_panel(self.modified_frame)
        algorithm_methods = {
            list(self.operations.values())[0][0]: self.apply_threshold,
            list(self.operations.values())[1][0]: self.apply_simple_halftone,
            list(self.operations.values())[1][1]: self.error_diffusion_halftoning,
            list(self.operations.values())[2][0]: self.convert_to_grayscale,
            list(self.operations.values())[2][1]: self.equalization_histogram,
            list(self.operations.values())[3][0]: self.sobel_simple_edge_detection,
            list(self.operations.values())[3][1]: self.prewitt_simple_edge_detection,
            list(self.operations.values())[3][2]: self.kirsch_edge_detection,
            list(self.operations.values())[4][0]: self.homogeneity_advanced_edge_detection,
            list(self.operations.values())[4][1]: self.differnce_op,
            list(self.operations.values())[4][2]: self.Difference_of_Gaussians,
            list(self.operations.values())[4][3]: self.contrast_based_edge_detection,
            list(self.operations.values())[4][4]: self.variance_advanced_edge_detection,
            list(self.operations.values())[4][5]: self.range_advanced_edge_detection,
            list(self.operations.values())[5][0]: self.high_filter,
            list(self.operations.values())[5][1]: self.low_filter,
            list(self.operations.values())[5][2]: self.median_filter,
            list(self.operations.values())[6][0]: self.invert_image,
            list(self.operations.values())[6][1]: self.add_image_and_copy,
            list(self.operations.values())[6][2]: self.subtract_image_and_copy,
            list(self.operations.values())[7][0]: self.manual_technique,
            list(self.operations.values())[7][1]: self.histogram_peak_technique,
            list(self.operations.values())[7][2]: self.valley_technique,
            list(self.operations.values())[7][3]: self.adaptive_technique,

        }
        
        algorithm_method = algorithm_methods.get(selected_algorithm)
        if algorithm_method:
            self.modified_frame.config(text=f"{selected_algorithm}")
            algorithm_method()
            if selected_algorithm == list(self.operations.values())[2][1]:
                self.display(self.custom_histogram(),"Histogram Equalization")
            img = ImageTk.PhotoImage(image=Image.fromarray(self.modified_img))
            self.modified_frame.pack(side="right",fill=tk.X,expand=1)
            self.modified_panel = tk.Label(self.modified_frame, image=img)
            self.modified_panel.image = img  # Keep a reference to avoid garbage collection
            self.modified_panel.pack(fill=tk.BOTH, expand=True)
        else:
            messagebox.showerror("Error", "Selected algorithm is not implemented.")
            
    
    def convert_to_grayscale(self):
        
        if self.original_img is not None:
            # Check if the image is already grayscale (single channel)
            if len(self.original_img.shape) == 3:  # Color image (3 channels)
                self.grey_img=cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            else:  # Image is already grayscale (1 channel)
                self.grey_img = self.original_img
    
    def threshold_image(self,th):

        thresholded_im = np.zeros(self.modified_img.shape)
        thresholded_im[self.modified_img >= th] = 255

        return thresholded_im

    def compute_otsu_criteria(self, th):
        thresholded_im = self.threshold_image(th)
        
        nb_pixels = self.modified_img.size
        nb_pixels1 = np.count_nonzero(thresholded_im)
        weight1 = nb_pixels1 / nb_pixels
        weight0 = 1 - weight1
        
        if weight1 == 0 or weight0 == 0:
            return np.inf
        
        val_pixels0 = self.modified_img[thresholded_im == 0]
        val_pixels1 = self.modified_img[thresholded_im == 255]
        
        var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
        var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
        
        return weight0 * var0+ weight1 * var1

    def find_best_threshold(self):
        threshold_range = range(np.max(self.modified_img)+1)
        criterias = [self.compute_otsu_criteria(th) for th in threshold_range]
        best_threshold = threshold_range[np.argmin(criterias)]
        
        return best_threshold

    def calcthreshold(self):
        self.modified_img=self.grey_img
        threshold = self.find_best_threshold()
        return threshold
    
    # def calcthreshold(self):
    #     self.modified_img=self.grey_img
    #     total_pixels = self.modified_img.shape[0] * self.modified_img.shape[1] # Total number of pixels
    #     total_intensity = 0
    
    #     for row in self.modified_img:
    #       for pixel in row:
    #           total_intensity += pixel
    
    #     threshold = total_intensity / total_pixels
    #     return threshold.astype(np.uint8)
    
    def apply_threshold(self):
        threshold=self.calcthreshold()
        self.modified_img = np.zeros_like(self.grey_img, dtype=np.uint8)
    
        # Apply the binary threshold
        self.modified_img[self.grey_img >threshold ] = 255
            

    def apply_simple_halftone(self):
        width, height = self.grey_img.shape
        block_size=5
        self.modified_img = np.zeros((width, height))

        for x in range(0, width, block_size):
            for y in range(0, height, block_size):
                block = self.grey_img[x:x + block_size, y:y + block_size]
                mean_value = np.mean(block)
                halftone_value = (mean_value > 128) * 255
                self.modified_img[x:x + block_size, y:y + block_size] = halftone_value
    
        
    def error_diffusion_halftoning(self):
        width, height = self.grey_img.shape
        self.modified_img= np.zeros_like(self.grey_img)
        # Error diffusion kernel
        diffusion_kernel = [
            (0, 1, 7 / 16),  # right
            (1, -1, 3 / 16), # bottom left
            (1, 0, 5 / 16),  # bottom
            (1, 1, 1 / 16)   # bottom right
        ]
    
        for x in range(width):
            for y in range(height):
                old_pixel = self.grey_img[x, y]
                new_pixel = 255 if old_pixel > 128 else 0
                
                self.modified_img[x, y] = new_pixel
                quant_error = old_pixel - new_pixel
                
                for dx, dy, weight in diffusion_kernel:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        self.modified_img[nx, ny] += quant_error * weight
    
    def custom_histogram(self):
        row,col= self.modified_img.shape
        hist=np.zeros(256, dtype=int)
    
        for i in range(row):
            for j in range(col):
                c=self.modified_img[i,j]
                hist[c]+=1
                
        return hist
    
    def equalization_histogram(self):
        row,col=self.grey_img.shape
        area= row*col
        self.modified_img=self.grey_img
        hist= self.custom_histogram()
        equalHist= np.zeros_like(hist,dtype=int)
    
        histSum=0
        for i in range(len(hist)):
            equalHist[i]=histSum
            histSum+=hist[i]
    
        self.modified_img=np.zeros_like(self.grey_img,dtype=int)
    
        for i in range(row):
            for j in range(col):
                v=self.grey_img[i,j]
                self.modified_img[i,j]=(equalHist[v]*(255/area))
        self.modified_img = np.clip(self.modified_img, 0, 255).astype(np.uint8)
        
    def sobel_simple_edge_detection(self):
        # self.grey_img = cv2.GaussianBlur(self.grey_img, (9, 9), 0)
        kernel_x = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]])

        kernel_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]])
        gradient_x = cv2.filter2D(self.grey_img, -1, kernel_x)
        gradient_y = cv2.filter2D(self.grey_img, -1, kernel_y)
        gradient_magnitude=cv2.magnitude(gradient_x.astype(np.float32), gradient_y.astype(np.float32))
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        self.modified_img=gradient_magnitude

    def prewitt_simple_edge_detection(self):
        prewitt_x = np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]])

        prewitt_y = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]])
        
        gradient_x = cv2.filter2D(self.grey_img, -1, prewitt_x)
        gradient_y = cv2.filter2D(self.grey_img, -1, prewitt_y)
        
        gradient_magnitude=cv2.magnitude(gradient_x.astype(np.float32), gradient_y.astype(np.float32))
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        self.modified_img=gradient_magnitude
        
        
    def kirsch_edge_detection(self):
        image = self.grey_img.astype('float64')
        
        # Define Kirsch masks
        kirsch_masks = [
            np.array([[5,  5,  5],  # north
                      [-3,  0, -3],
                      [-3, -3, -3]]),
            np.array([[5,  5, -3],  # northeast
                      [5,  0, -3],
                      [-3, -3, -3]]),
            np.array([[5, -3, -3],  # east
                      [5,  0, -3],
                      [5, -3, -3]]),
            np.array([[-3, -3, -3],  # southeast
                      [5,  0, -3],
                      [5,  5, -3]]),
            np.array([[-3, -3, -3],  # south
                      [-3,  0, -3],
                      [5,  5,  5]]),
            np.array([[-3, -3, -3],  # southwest
                      [-3,  0,  5],
                      [-3,  5,  5]]),
            np.array([[-3, -3,  5],  # west
                      [-3,  0,  5],
                      [-3, -3,  5]]),
            np.array([[-3,  5,  5],  # northwest
                      [-3,  0,  5],
                      [-3, -3, -3]])
        ]
        
        
        responses = [cv2.filter2D(image, -1, mask) for mask in kirsch_masks]
        global_max_response = -np.inf
        global_direction = -1
        for i, response in enumerate(responses):
            max_response = np.max(response)  # Maximum response for this mask
            if max_response > global_max_response:
                global_max_response = max_response
                global_direction = i  # Index of the mask producing the global maximum response
        direction_label=""
        if global_direction == 0:
            direction_label = "NORTH"
        elif global_direction == 1:
            direction_label = "northeast"
        elif global_direction == 2:
            direction_label = "east"
        elif global_direction == 3:
            direction_label = "southeast"
        elif global_direction == 4:
            direction_label = "south"
        elif global_direction == 5:
            direction_label = "southwest"
        elif global_direction == 6:
            direction_label = "west"
        elif global_direction == 7:
            direction_label = "northwest"
        
        
        self.modified_frame.config(text=f"Kirch with Direction : {direction_label.upper()} ")
        self.modified_img = responses[global_direction]
      
    def homogeneity_advanced_edge_detection(self):
        height, width = self.grey_img.shape
        threshold = self.calcthreshold() 
        self.modified_img = np.zeros_like(self.grey_img, dtype=np.uint8)  # Ensure output is uint8
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                cPixel = self.grey_img[i, j] 
                Diff = [
                    abs(cPixel - int(self.grey_img[i - 1, j - 1])),
                    abs(cPixel - int(self.grey_img[i - 1, j])),
                    abs(cPixel - int(self.grey_img[i - 1, j + 1])),
                    abs(cPixel - int(self.grey_img[i, j - 1])),
                    abs(cPixel - int(self.grey_img[i, j + 1])),
                    abs(cPixel - int(self.grey_img[i + 1, j - 1])),
                    abs(cPixel - int(self.grey_img[i + 1, j])),
                    abs(cPixel - int(self.grey_img[i + 1, j + 1]))
                ]
                homoResult = max(Diff)
                # Apply threshold and ensure the result fits in uint8
                self.modified_img[i, j] = 255 if homoResult >= threshold else 0
    
    def differnce_op(self):
        height, width = self.grey_img.shape
        threshold = self.calcthreshold() 
        self.modified_img = np.zeros_like(self.grey_img, dtype=np.uint8)  # Ensure output is uint8
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                Diff = [
                    abs(int(self.grey_img[i + 1, j + 1]) - int(self.grey_img[i - 1, j - 1])),
                    abs(int(self.grey_img[i + 1, j])     - int(self.grey_img[i - 1, j])),
                    abs(int(self.grey_img[i + 1, j - 1]) - int(self.grey_img[i - 1, j + 1])),
                    abs(int(self.grey_img[i, j + 1]) - int(self.grey_img[i, j - 1]))
                ]
                differnce = max(Diff)
                # Apply threshold and ensure the result fits in uint8
                self.modified_img[i, j] = 255 if differnce >= threshold else 0
        
    def contrast_based_edge_detection(self):
        mask = np.array([
            [-1,0,-1],
            [0,4,0],
            [-1,0,-1]
        ], dtype=np.float32)
        quick_edge=cv2.filter2D(self.grey_img, -1, mask)
        # self.modified_img= quick_edge
        # Create an averaging filter to compute the local average (all ones)
        averaging_kernel = np.ones((3, 3), np.float32) / 9  # Normalized kernel
        local_avg = cv2.filter2D(self.grey_img, -1, averaging_kernel)  # Apply the averaging filter
        with np.errstate(divide='ignore', invalid='ignore'): # avoid division by 0
            smoothed_image_array = np.divide(quick_edge, local_avg)
            smoothed_image_array = np.nan_to_num(smoothed_image_array, nan=0.0, posinf=0.0, neginf=0.0)
        smoothed_image_array = cv2.normalize(smoothed_image_array, None, 0, 255, cv2.NORM_MINMAX)
     
        self.modified_img = smoothed_image_array.astype(np.uint8)
            
        
        
    def Difference_of_Gaussians(self):
        
        mask_7x7 = np.array([
            [0, 0, -1, -1, -1, 0, 0],
            [0, -2, -3, -3, -3, -2, 0],
            [-1, -3, 5, 5, 5, -3, -1],
            [-1, -3, 5, 16, 5, -3, -1],
            [-1, -3, 5, 5, 5, -3, -1],
            [0, -2, -3, -3, -3, -2, 0],
            [0, 0, -1, -1, -1, 0, 0]
        ], dtype=np.float32)
        
        mask_9x9 = np.array([
            [0, 0, 0, -1, -1, -1, 0, 0, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [-1, -3, -1, 9, 19, 9, -1, -3, -1],
            [-1, -3, -1, 9, 9, 9, -1, -3, -1],
            [0, -3, -2, -1, -1, -1, -2, -3, 0],
            [0, -2, -3, -3, -3, -3, -3, -2, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, 0]
        ], dtype=np.float32)
        # Apply convolution with 7x7 and 9x9 kernels using OpenCV
        filtered_7x7 = cv2.filter2D(self.grey_img, ddepth=-1, kernel=mask_7x7, borderType=cv2.BORDER_CONSTANT)
        filtered_9x9 = cv2.filter2D(self.grey_img, ddepth=-1, kernel=mask_9x9, borderType=cv2.BORDER_CONSTANT)
        self.modified_img=filtered_7x7 -filtered_9x9
        # self.modified_img=filtered_7x7 
    
    def variance_advanced_edge_detection(self):
        
        self.modified_img = np.zeros_like(self.grey_img, dtype=np.float32)
        height, width = self.grey_img.shape
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = self.grey_img[i-1:i+2, j-1:j+2]
                mean = np.mean(neighborhood)
                variance = np.sum((neighborhood - mean) ** 2) / 9
                self.modified_img[i, j] = variance
        
    def range_advanced_edge_detection(self):
        self.modified_img = np.zeros_like(self.grey_img, dtype=np.float32)
        height, width = self.grey_img.shape
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = self.grey_img[i-1:i+2, j-1:j+2]
                range_value = np.max(neighborhood) - np.min(neighborhood)
                self.modified_img[i, j] = range_value

    def high_filter(self):
        
        mask=np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]],dtype=np.float32)
    
        self.modified_img=cv2.filter2D(self.grey_img, -1 ,mask)
    
    def low_filter(self):
        mask=np.array([[0,1/6,0],
                       [1/6,2/6,1/6],
                       [0,1/6,0]],   dtype=np.float32) 
        
        self.modified_img=cv2.filter2D(self.grey_img, -1 ,mask)
    
    def median_filter(self):
        kernel_size=5
        # Apply the median filter
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        
        # Get the padding size
        pad_size = kernel_size // 2
        
        # Pad the image to handle edges
        padded_image = np.pad(self.grey_img, pad_size, mode='constant', constant_values=0)
        
        # Create an empty output image
        self.modified_img = np.zeros_like(self.grey_img)
        
        # Iterate over each pixel in the image
        for i in range(self.grey_img.shape[0]):
            for j in range(self.grey_img.shape[1]):
                # Extract the neighborhood window
                window = padded_image[i:i + kernel_size, j:j + kernel_size]
                
                # Compute the median value
                median_value = np.median(window)
                
                # Assign the median value to the output image
                self.modified_img[i, j] = median_value
    
    def add_image_and_copy(self):
        # # Perform pixel-wise addition
        
        # height, width = self.grey_img.shape
        # img_copy = self.grey_img.copy()
        # result = np.zeros_like(self.grey_img, dtype=np.uint8)
        
        # for y in range(height):
        #     for x in range(width):
        #         result[y, x] = min(int(self.grey_img[y, x]) + int(img_copy[y, x]), 255)
        
        # self.modified_img = result
        
        second_img_file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")])
        while(not second_img_file_path):
            
            messagebox.showwarning("No Image selected","Image must be uploaded ")
            second_img_file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    
        second_img = cv2.imread(second_img_file_path, cv2.IMREAD_GRAYSCALE)
        if second_img is None:
            second_img=self.grey_img
    
        if self.grey_img.shape != second_img.shape:
            second_img = cv2.resize(second_img, (self.grey_img.shape[1], self.grey_img.shape[0]))
        height, width = self.grey_img.shape
        result = np.zeros_like(self.grey_img, dtype=np.uint8)
    
        for y in range(height):
            for x in range(width):
                result[y, x] = min(int(self.grey_img[y, x]) + int(second_img[y, x]), 255)
    
        self.modified_img = result
        
    def subtract_image_and_copy(self):
        second_img_file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp;*.webp")])
        while(not second_img_file_path):
            
            messagebox.showwarning("No Image selected","Image must be uploaded ")
            second_img_file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp;*.webp")])
    
        second_img = cv2.imread(second_img_file_path, cv2.IMREAD_GRAYSCALE)
        if second_img is None:
            second_img=self.grey_img
    
        if self.grey_img.shape != second_img.shape:
            second_img = cv2.resize(second_img, (self.grey_img.shape[1], self.grey_img.shape[0]))
        # Perform pixel-wise subtraction
        height, width = self.grey_img.shape
        result = np.zeros_like(self.grey_img, dtype=np.uint8)
    
        for y in range(height):
            for x in range(width):
                result[y, x] = max(int(self.grey_img[y, x]) - int(second_img[y, x]), 0)
    
        self.modified_img = result

        
    def invert_image(self):

        height,width = self.grey_img.shape
        self.modified_img = np.zeros_like(self.grey_img, dtype=np.uint8)
       
        for i in range(height):
           for j in range(width):
               self.modified_img[i,j]=255-self.grey_img[i,j]
     
    def peaks_high_low(self,histogram, peak1, peak2):
        midpoint = (peak1 + peak2) // 2
        
        if peak1>peak2:
            peak1,peak2 = peak2,peak1
        low =  midpoint
        high = peak2
      
        return low, high
        
    def smooth_histogram(self,histogram):
        smoothed_hist = np.zeros_like(histogram)
        smoothed_hist[0] = (histogram[0] + histogram[1]) / 2
        smoothed_hist[-1] = (histogram[-1] + histogram[-2]) / 2
      
        for i in range(1, len(histogram) - 1):
            smoothed_hist[i] = (histogram[i-1] + histogram[i] + histogram[i+1]) / 3
      
        return smoothed_hist
    
    def calc_peaks(self,hist,th):
        backgroud_peak=np.argmax(hist[:th])
        object_peak=th + np.argmax(hist[th:])
        return backgroud_peak,object_peak
    
    def manual_technique(self):
        
        self.modified_img= np.zeros_like(self.grey_img)
        self.modified_img[ (self.grey_img>=200) & (self.grey_img<=242) ]=255
    
    def histogram_peak_technique(self):
        threshold=self.calcthreshold()
        hist=self.custom_histogram()
        smoothed_hist =self.smooth_histogram(hist)
        peak1,peak2=self.calc_peaks(smoothed_hist ,threshold)
        low,high = self.peaks_high_low(smoothed_hist ,peak1,peak2)
        if low > high:
            low,high=high,low
        self.modified_img= np.zeros_like(self.grey_img)
        self.modified_img[ (self.grey_img>=low) & (self.grey_img<=high) ]= 255
   
    def valley_technique(self):
        threshold=self.calcthreshold()
        hist=self.custom_histogram()
        smoothed_hist =self.smooth_histogram(hist)
        peak1,peak2=self.calc_peaks(smoothed_hist ,threshold)
        valley_point = self.valley_point(smoothed_hist, peak1, peak2)
        low,high = valley_point,peak2
        if low > high:
            low,high=high,low
        self.modified_img= np.zeros_like(self.grey_img)
        self.modified_img[ (self.grey_img>=low) & (self.grey_img<=high) ]= 255
    
    def valley_point(self,histogram, peak1, peak2):
        if peak1>peak2:
            peak1,peak2 = peak2,peak1
        vall_point= peak1 + np.argmin(histogram[peak1:peak2])
        return vall_point
    
    def adaptive_technique(self):
        threshold=self.calcthreshold()
        hist=self.custom_histogram()
        smoothed_hist =self.smooth_histogram(hist)
        
        #First pass 
        peak1,peak2=self.calc_peaks(smoothed_hist ,threshold)
        low,high = self.peaks_high_low(smoothed_hist ,peak1,peak2)
        
        #Second pass 
        back,obj=self.threshold_and_means(low,high)
        low,high = self.peaks_high_low(smoothed_hist ,back,obj)
        low=low.astype(np.uint8)
        high=high.astype(np.uint8)
        
        if low > high:
            low,high=high,low
        self.modified_img= np.zeros_like(self.grey_img)
        self.modified_img[ (self.grey_img>=low) & (self.grey_img<=high) ]= 255
        
    def threshold_and_means(self,low,high):
        rows, cols = self.grey_img.shape
        mask = (low <= self.grey_img) & (self.grey_img <= high)
        self.modified_img = np.where(mask, 255, 0)
      
        object_pixels = self.grey_img[mask]
        background_pixels = self.grey_img[~mask]
      
        object_mean = np.mean(object_pixels)
        background_mean = np.mean(background_pixels)
        
        return background_mean,object_mean 
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
