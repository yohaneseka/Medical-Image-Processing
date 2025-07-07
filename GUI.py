import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
from skimage import color, filters
from skimage.color import label2rgb, rgb2gray
from io import BytesIO
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label  # Correct import
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from skimage import exposure,filters, morphology, segmentation, feature, measure, color
import scipy.ndimage as ndi
from skimage.morphology import binary_dilation, disk, binary_closing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.io import imread
import os
import cv2
import imageio.v2 as imageio

st.set_page_config(
    page_title="All Of Medical Image Processing Assignments",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Medical Image Processing")

task_choice = st.sidebar.radio("Select Task", ["Task 1: Original Methods", "Task 2: Gaussian and Sharpening", "Task 3: Corner, Line, and Circle Detection", "Final Project"])

if task_choice == "Task 1: Original Methods":
    # Settings for Task 1 - No tabs, directly in main area
    st.header("Settings")
    uploaded_file_task1 = st.file_uploader("Upload an image for Task 1", type=["jpg", "jpeg", "png"], key="task1_uploader")
    
    st.subheader("Select Edge Detection Methods")
    col1, col2, col3 = st.columns(3)
    with col1:
        prewitt_enabled = st.checkbox("Prewitt", value=True)
        sobel_enabled = st.checkbox("Sobel", value=True)
    with col2:
        roberts_enabled = st.checkbox("Roberts", value=True)
        extended_sobel_enabled = st.checkbox("Extended Sobel", value=True)
    with col3:
        kirsch_enabled = st.checkbox("Kirsch", value=True)
    
    st.subheader("Analysis Options")
    show_opts_col1, show_opts_col2 = st.columns(2)
    with show_opts_col1:
        show_histograms = st.checkbox("Show Histograms", value=True)
        show_timing = st.checkbox("Show Timing Comparison", value=True)
    with show_opts_col2:
        show_metrics = st.checkbox("Show Statistical Metrics", value=True)
        
    # Add a separator between settings and results
    st.markdown("---")

    # Task 1 Functions
    def apply_prewitt(image):
        Gx_Prewitt = np.array([[-1.0, 0.0, 1.0], 
                            [-1.0, 0.0, 1.0], 
                            [-1.0, 0.0, 1.0]])
        
        Gy_Prewitt = np.array([[-1.0, -1.0, -1.0], 
                            [0.0, 0.0, 0.0], 
                            [1.0, 1.0, 1.0]])
        
        x_prewitt = cv2.filter2D(image, -1, Gx_Prewitt)
        y_prewitt = cv2.filter2D(image, -1, Gy_Prewitt)
        
        prewitt_image = np.sqrt(x_prewitt**2 + y_prewitt**2)
        
        return prewitt_image

    def apply_sobel(image):
        Gx_Sobel = np.array([[-1.0, 0.0, 1.0], 
                        [-2.0, 0.0, 2.0], 
                        [-1.0, 0.0, 1.0]])
        
        Gy_Sobel = np.array([[-1.0, -2.0, -1.0], 
                        [0.0, 0.0, 0.0], 
                        [1.0, 2.0, 1.0]])
        
        x_sobel = cv2.filter2D(image, -1, Gx_Sobel)
        y_sobel = cv2.filter2D(image, -1, Gy_Sobel)
        
        sobel_image = np.sqrt(x_sobel**2 + y_sobel**2)
        
        return sobel_image

    def apply_roberts(image):
        H1 = np.array([[1.0, 0.0], 
                    [0.0, -1.0]])
        
        H2 = np.array([[0.0, 1.0], 
                    [-1.0, 0.0]])
        
        H1_Roberts = cv2.filter2D(image, -1, H1)
        H2_Roberts = cv2.filter2D(image, -1, H2)
        
        roberts_image = np.sqrt(H1_Roberts**2 + H2_Roberts**2)
        
        return roberts_image

    def extended_sobel(image):
        if len(image.shape) > 2:
            raise Exception("Illegal argument: input must be a single channel image (gray)")

        HES0 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        HES1 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
        HES2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        HES3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])

        D0 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, HES0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        D1 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, HES1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        D2 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, HES2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        D3 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, HES3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        magn_ES = cv2.max(D0, cv2.max(D1, cv2.max(D2, cv2.max(-D0, cv2.max(-D1, cv2.max(-D2, -D3))))))
        
        return magn_ES

    def kirsch_filter(image):
        if len(image.shape) > 2:
            raise Exception("Illegal argument: input must be a single channel image (gray)")

        H0 = np.array([[-5, 3, 3], [-5, 0, 3], [-5, 3, 3]])
        H1 = np.array([[-5, -5, 3], [-5, 0, 3], [3, 3, 3]])
        H2 = np.array([[-5, -5, -5], [3, 0, 3], [3, 3, 3]])
        H3 = np.array([[3, -5, -5], [3, 0, -5], [3, 3, 3]])

        g0 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, H0), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        g1 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, H1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        g2 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, H2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        g3 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, H3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        magn = cv2.max(g0, cv2.max(g1, cv2.max(g2, cv2.max(-g0, cv2.max(-g1, cv2.max(-g2, -g3))))))
        
        return magn

    def process_image(image):
        # Convert image to grayscale if it's color
        if len(image.shape) > 2:
            gray_image = color.rgb2gray(image)
            gray_image = (gray_image * 255).astype(np.uint8)
        else:
            gray_image = image
        
        # Store results and timing
        results = {}
        timing_results = {}
        
        # Configure detection methods based on user selection
        detection_methods = {}
        if prewitt_enabled:
            detection_methods['Prewitt'] = apply_prewitt
        if sobel_enabled:
            detection_methods['Sobel'] = apply_sobel
        if roberts_enabled:
            detection_methods['Roberts'] = apply_roberts
        if extended_sobel_enabled:
            detection_methods['Extended Sobel'] = extended_sobel
        if kirsch_enabled:
            detection_methods['Kirsch'] = kirsch_filter
        
        # Apply selected methods
        for method_name, method_func in detection_methods.items():
            start_time = time.time()
            results[method_name] = method_func(gray_image)
            end_time = time.time()
            timing_results[method_name] = (end_time - start_time)
        
        return gray_image, results, timing_results
    
    # Results - Main application logic for Task 1 (directly after settings)
    if uploaded_file_task1 is not None:
            # Read and display the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file_task1.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
            # Check if any method is selected
            if any([prewitt_enabled, sobel_enabled, roberts_enabled, extended_sobel_enabled, kirsch_enabled]):
                # Process the image
                gray_image, results, timing_results = process_image(image)
                
                # Display results
                st.subheader("Edge Detection Results")
                
                # Create columns for each selected method
                cols = st.columns(len(results) + 1)
                
                # Display original grayscale
                with cols[0]:
                    st.markdown("**Grayscale Original**")
                    st.image(gray_image, use_container_width=True)
                
                # Display results for each method
                for i, (method_name, result) in enumerate(results.items(), 1):
                    with cols[i]:
                        st.markdown(f"**{method_name}**")
                        # Normalize result for display
                        result_normalized = (result - result.min()) / (result.max() - result.min()) * 255
                        result_normalized = result_normalized.astype(np.uint8)
                        st.image(result_normalized, use_container_width=True)
                        st.caption(f"Time: {timing_results[method_name]*1000:.2f} ms")
                
                # Statistical metrics
                if show_metrics:
                    st.subheader("Statistical Metrics")
                    
                    # Create a DataFrame for metrics
                    metrics_data = []
                    for method_name, result in results.items():
                        metrics_data.append({
                            "Method": method_name,
                            "Min": f"{result.min():.2f}",
                            "Max": f"{result.max():.2f}",
                            "Mean": f"{result.mean():.2f}",
                            "Execution Time (ms)": f"{timing_results[method_name]*1000:.2f}"
                        })
                    
                    # Display metrics as a table
                    st.table(metrics_data)
                
                # Histograms
                if show_histograms:
                    st.subheader("Histograms")
                    
                    fig_hist = plt.figure(figsize=(15, 10))
                    
                    # Original image histogram
                    plt.subplot(2, 3, 1)
                    plt.hist(gray_image.ravel(), bins=256, range=(0, 256))
                    plt.title('Original Image Histogram')
                    
                    # Method histograms
                    positions = {}
                    for i, method_name in enumerate(results.keys(), 2):
                        positions[method_name] = i
                    
                    for method_name, position in positions.items():
                        plt.subplot(2, 3, position)
                        plt.hist(results[method_name].ravel(), bins=50)
                        plt.title(f'{method_name} Histogram')
                    
                    plt.tight_layout()
                    st.pyplot(fig_hist)
                
                # Timing comparison
                if show_timing and len(timing_results) > 1:
                    st.subheader("Execution Time Comparison")
                    
                    fig_time = plt.figure(figsize=(10, 6))
                    methods = list(timing_results.keys())
                    times = [timing_results[method]*1000 for method in methods]
                    
                    plt.bar(methods, times)
                    plt.ylabel('Execution Time (ms)')
                    plt.title('Edge Detection Methods - Runtime Comparison')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig_time)
            else:
                st.warning("Please select at least one edge detection method.")
    else:
            # Display sample message when no image is uploaded
        st.info("Please upload an image to start the analysis.")

# ===================== TASK 2: ADVANCED ANALYSIS =====================
elif task_choice == "Task 2: Gaussian and Sharpening":
    st.header("Task 2: Gaussian and Sharpening")
    
    # Task 2 settings
    st.subheader("Canny Edge Detection Pipeline Settings")
    uploaded_file_task2 = st.file_uploader("Upload an image for advanced analysis", type=["jpg", "jpeg", "png"], key="task2_uploader")

    tab1, tab2 = st.tabs(["Canny Edge Detection", "Sharpening"])  

    with tab1:
            st.subheader("Canny Edge Detection") 
            
            # Add implementation selection
            implementation_type = st.radio(
                "Implementation Method:",
                ["Custom Implementation", "Library Implementation", "Comparison (Both)"],
                horizontal=True
            )
            
            # Create columns for parameters
            col1, col2, col3 = st.columns(3)
            
            # Custom implementation parameters
            with col1:
                st.write("**Custom Implementation Parameters:**")
                sigma = st.slider("Gaussian Blur Sigma", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                help="Controls the amount of blur. Higher values create more blur.")
                kernel_size = st.slider("Kernel Size", min_value=3, max_value=15, value=5, step=2,
                                help="Size of the Gaussian kernel. Must be odd number.")
                                
            with col2:
                st.write("**Custom Threshold Parameters:**")
                thlo = st.slider("Low Threshold", min_value=1, max_value=50, value=10, step=1,
                            help="Low threshold for edge detection. Lower values include more potential edges.")
                thhi = st.slider("High Threshold", min_value=10, max_value=100, value=23, step=1,
                                        help="High threshold for edge detection. Higher values detect fewer edges.")
            
            with col3:
                st.write("**Library Implementation Parameters:**")
                lib_sigma = st.slider("Library Gaussian Sigma", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                            help="Sigma parameter for library implementation")
                lib_low = st.slider("Library Low Threshold", min_value=10, max_value=150, value=50, step=5,
                            help="Low threshold for Canny edge detector library implementation")
                lib_high = st.slider("Library High Threshold", min_value=50, max_value=250, value=150, step=5,
                            help="High threshold for Canny edge detector library implementation")
                lib_aperture = st.selectbox("Library Aperture Size", [3, 5, 7], index=0,
                                help="Aperture size for the Sobel operator in library implementation")
            
            # Additional parameters and display options
            T = st.slider("Gaussian Truncation (T)", min_value=0.01, max_value=0.5, value=0.3, step=0.01,
                        help="Truncation factor for Gaussian kernel generation")
            show_all_steps = st.checkbox("Show All Processing Steps", value=True,
                                    help="Display every step of the Canny edge detection pipeline")
            
            # Separator between settings and results
            st.markdown("---")
            
            # Define all utility functions
            def sHalf(T, sigma):
                temp = -np.log(T) * 2 * (sigma**2)
                return np.round(np.sqrt(temp))

            def calculate_filter_size(T, sigma):
                return 2 * sHalf(T, sigma) + 1

            def maskGeneration(T, sigma):
                N = calculate_filter_size(T, sigma)
                shalf = sHalf(T, sigma)
                y, x = np.meshgrid(range(-int(shalf), int(shalf) + 1), 
                                range(-int(shalf), int(shalf) + 1))
                return x, y

            def Gaussian(x, y, sigma):
                temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
                gaussian_kernel = np.exp(-temp)
                
                # Normalize kernel
                return gaussian_kernel / gaussian_kernel.sum()

            def gaussian_kernel(size, sigma=1.0):
                size = int(size) // 2
                x, y = np.mgrid[-size:size+1, -size:size+1]
                normal = 1 / (2.0 * np.pi * sigma**2)
                g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
                return g / g.sum()

            def pad(img, kernel):
                r, c = img.shape
                kr, kc = kernel.shape
                padded = np.zeros((r + kr - 1, c + kc - 1), dtype=img.dtype)
                insert = np.uint((kr)//2)
                padded[int(insert):int(insert + r), int(insert):int(insert + c)] = img
                return padded

            def smooth(img, kernel=None):
                """Smoothing image with kernel"""
                if kernel is None:
                    mask = np.array([[1,1,1],[1,1,1],[1,1,1]])
                else:
                    mask = kernel
                i, j = mask.shape
                output = np.zeros((img.shape[0], img.shape[1]))
                image_padded = pad(img, mask)
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        output[x, y] = (mask * image_padded[x:x+i, y:y+j]).sum() / mask.sum()
                return output

            def calculate_gradient_X(x, y, sigma):
                temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
                return -(x * np.exp(-temp)) / (sigma ** 2)

            def calculate_gradient_Y(x, y, sigma):
                temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
                return -(y * np.exp(-temp)) / (sigma ** 2)

            def create_Gx(fx, fy):
                sigma = 1.0
                gx = calculate_gradient_X(fx, fy, sigma)
                gx = (gx * 255)
                return np.around(gx)

            def create_Gy(fx, fy):
                sigma = 1.0
                gy = calculate_gradient_Y(fx, fy, sigma)
                gy = (gy * 255)
                return np.around(gy)

            def ApplyMask(image, kernel):
                i, j = kernel.shape
                kernel = np.flipud(np.fliplr(kernel))
                output = np.zeros_like(image)
                image_padded = pad(image, kernel)
                for x in range(image.shape[0]):
                    for y in range(image.shape[1]):
                        output[x, y] = (kernel * image_padded[x:x+i, y:y+j]).sum()
                return output

            def calculate_magnitude(fx, fy):
                mag = np.sqrt((fx ** 2) + (fy ** 2))
                mag = mag * 100 / mag.max() if mag.max() > 0 else mag
                return np.around(mag)

            def Gradient_Direction(fx, fy):
                return np.rad2deg(np.arctan2(fy, fx)) + 180

            def Digitize_angle(angle):
                quantized = np.zeros((angle.shape[0], angle.shape[1]))
                for i in range(angle.shape[0]):
                    for j in range(angle.shape[1]):
                        # 0 degrees (yellow)
                        if (0 <= angle[i, j] < 22.5 or 157.5 <= angle[i, j] < 202.5 or 337.5 <= angle[i, j] <= 360):
                            quantized[i, j] = 0
                        # 45 degrees (green)
                        elif (22.5 <= angle[i, j] < 67.5 or 202.5 <= angle[i, j] < 247.5):
                            quantized[i, j] = 1
                        # 90 degrees (red)
                        elif (67.5 <= angle[i, j] < 112.5 or 247.5 <= angle[i, j] < 292.5):
                            quantized[i, j] = 2
                        # 135 degrees (blue)
                        elif (112.5 <= angle[i, j] < 157.5 or 292.5 <= angle[i, j] < 337.5):
                            quantized[i, j] = 3
                return quantized

            def Non_Max_Supp(qn, magn, D):
                M = np.zeros(qn.shape)
                a, b = np.shape(qn)
                for i in range(1, a-1):  # Avoid index out of bounds
                    for j in range(1, b-1):  # Avoid index out of bounds
                        if qn[i, j] == 0:
                            if magn[i, j-1] < magn[i, j] or magn[i, j] > magn[i, j+1]:
                                M[i, j] = D[i, j]
                            else:
                                M[i, j] = 0
                        elif qn[i, j] == 1:
                            if magn[i-1, j+1] <= magn[i, j] or magn[i, j] >= magn[i+1, j-1]:
                                M[i, j] = D[i, j]
                            else:
                                M[i, j] = 0
                        elif qn[i, j] == 2:
                            if magn[i-1, j] <= magn[i, j] or magn[i, j] >= magn[i+1, j]:
                                M[i, j] = D[i, j]
                            else:
                                M[i, j] = 0
                        elif qn[i, j] == 3:
                            if magn[i-1, j-1] <= magn[i, j] or magn[i, j] >= magn[i+1, j+1]:
                                M[i, j] = D[i, j]
                            else:
                                M[i, j] = 0
                return M

            def double_thresholding(g_supp, thlo, thhi):
                g_thresholded = np.zeros(g_supp.shape)
                for i in range(0, g_supp.shape[0]):
                    for j in range(0, g_supp.shape[1]):
                        if g_supp[i, j] < thlo:
                            g_thresholded[i, j] = 0
                        elif g_supp[i, j] >= thlo and g_supp[i, j] < thhi:
                            g_thresholded[i, j] = 128  # weak edge
                        else:
                            g_thresholded[i, j] = 255  # strong edge > thhi
                return g_thresholded

            def hysterisis(g_thresholded):
                g_strong = np.zeros(g_thresholded.shape)
                for i in range(1, g_thresholded.shape[0]-1):  # Avoid index out of bounds
                    for j in range(1, g_thresholded.shape[1]-1):  # Avoid index out of bounds
                        val = g_thresholded[i, j]
                        if val == 128:
                            # Check if there's a strong edge in neighbors
                            if (g_thresholded[i-1, j] == 255 or g_thresholded[i+1, j] == 255 or 
                                g_thresholded[i-1, j-1] == 255 or g_thresholded[i+1, j+1] == 255 or
                                g_thresholded[i, j-1] == 255 or g_thresholded[i, j+1] == 255 or
                                g_thresholded[i-1, j+1] == 255 or g_thresholded[i+1, j-1] == 255):
                                g_strong[i, j] = 255  # Becomes strong edge
                        elif val == 255:
                            g_strong[i, j] = 255
                return g_strong

            def color(quantized, mag):
                color_img = np.zeros((mag.shape[0], mag.shape[1], 3), np.uint8)
                a, b = np.shape(mag)
                for i in range(a-1):
                    for j in range(b-1):
                        if quantized[i,j] == 0:
                            if mag[i,j] != 0:
                                color_img[i,j,0] = 255
                            else:
                                color_img[i,j,0] = 0
                                
                        elif quantized[i,j] == 1:
                            if mag[i,j] != 0:
                                color_img[i,j,1] = 255  # Green
                            else:
                                color_img[i,j,1] = 0
                                
                        elif quantized[i,j] == 2:
                            if mag[i,j] != 0:
                                color_img[i,j,2] = 255  # Red
                            else:
                                color_img[i,j,2] = 0
                                
                        elif quantized[i,j] == 3:
                            if mag[i,j] != 0:
                                color_img[i,j,0] = 255
                                color_img[i,j,1] = 255
                            else:
                                color_img[i,j,0] = 0
                                color_img[i,j,1] = 0
                return color_img

            def apply_canny_pipeline(image):
                start_time = time.time()
                
                # Convert image to grayscale if it's color
                if len(image.shape) > 2:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = image
                
                # Step 1: Create Gaussian kernel for smoothing
                gauss = gaussian_kernel(kernel_size, sigma)
                
                # Step 2: Smooth image with Gaussian filter
                smooth_img = smooth(gray_image, gauss)
                
                # Step 3: Calculate x and y gradients
                size = kernel_size
                x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
                
                # Create gradient filters
                gx = create_Gx(x, y)
                gy = create_Gy(x, y)
                
                # Apply filters to smoothed image
                fx = ApplyMask(smooth_img, gx)
                fy = ApplyMask(smooth_img, gy)
                
                # Step 4: Calculate magnitude, gradient angle and quantize angle
                mag = calculate_magnitude(fx, fy)
                angle = Gradient_Direction(fx, fy)
                quantized = Digitize_angle(angle)
                
                # Step 5: Non-maximum suppression
                non_max_img = Non_Max_Supp(quantized, mag, mag)
                
                # Step 6: Double threshold
                threshold_img = double_thresholding(non_max_img, thlo, thhi)
                
                # Step 7: Hysteresis - connect weak edges to strong edges
                hysteresis_img = hysterisis(threshold_img)
                
                # Generate color-coded edge direction visualization
                colored_edges = color(quantized, non_max_img)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Convert arrays to 8-bit for display
                mag_normalized = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else np.zeros_like(mag, dtype=np.uint8)
                fx_normalized = ((fx - fx.min()) / (fx.max() - fx.min()) * 255).astype(np.uint8) if fx.max() > fx.min() else np.zeros_like(fx, dtype=np.uint8)
                fy_normalized = ((fy - fy.min()) / (fy.max() - fy.min()) * 255).astype(np.uint8) if fy.max() > fy.min() else np.zeros_like(fy, dtype=np.uint8)
                
                return {
                    'gray_image': gray_image,
                    'gaussian_image': smooth_img.astype(np.uint8),
                    'fx': fx_normalized,
                    'fy': fy_normalized,
                    'magnitude': mag_normalized,
                    'angle': angle,
                    'color_edges': colored_edges,
                    'non_max_img': non_max_img.astype(np.uint8),
                    'threshold_img': threshold_img.astype(np.uint8),
                    'hysteresis_img': hysteresis_img.astype(np.uint8),
                    'execution_time': execution_time
                }
            
            # Function to run the library implementation of Canny
            def apply_library_canny(image):
                start_time = time.time()
                
                # Convert image to grayscale if it's color
                if len(image.shape) > 2:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = image
                    
                # Step 1: Apply Gaussian blur (library implementation)
                gaussian_image = cv2.GaussianBlur(gray_image, (0, 0), lib_sigma)
                
                # Step 2: Calculate gradients using Sobel (library implementation)
                fx = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0, ksize=lib_aperture)
                fy = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1, ksize=lib_aperture)
                
                # Step 3: Calculate magnitude and angle
                magnitude = np.sqrt(fx**2 + fy**2)
                magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8) if magnitude.max() > 0 else magnitude
                angle = np.arctan2(fy, fx) * 180 / np.pi
                
                # Step 4: Apply Canny edge detection (library implementation)
                canny_edges = cv2.Canny(gaussian_image, lib_low, lib_high, apertureSize=lib_aperture)
                
                # Create color visualization for edges based on direction
                hsv = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
                hsv[..., 0] = ((angle + 180) / 360 * 180).astype(np.uint8)  # Hue from angle
                hsv[..., 1] = 255  # Full saturation
                hsv[..., 2] = np.minimum(magnitude, 255)  # Value from magnitude
                color_edges = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
                # Apply mask to only show colors where edges are detected
                for i in range(3):
                    color_edges[..., i] = color_edges[..., i] * (canny_edges > 0)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Normalize fx and fy for visualization
                fx_normalized = ((fx - fx.min()) / (fx.max() - fx.min()) * 255).astype(np.uint8) if fx.max() > fx.min() else np.zeros_like(fx, dtype=np.uint8)
                fy_normalized = ((fy - fy.min()) / (fy.max() - fy.min()) * 255).astype(np.uint8) if fy.max() > fy.min() else np.zeros_like(fy, dtype=np.uint8)
                
                return {
                    'gray_image': gray_image,
                    'gaussian_image': gaussian_image,
                    'fx': fx_normalized,
                    'fy': fy_normalized,
                    'magnitude': magnitude,
                    'angle': angle,
                    'canny_edges': canny_edges,
                    'color_edges': color_edges,
                    'execution_time': execution_time
                }
            
            # Advanced Results area
            if uploaded_file_task2 is not None:
                try:
                    # Read the uploaded image
                    file_bytes = np.asarray(bytearray(uploaded_file_task2.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Reset file pointer for potential reuse in other tabs
                    uploaded_file_task2.seek(0)
                    
                    if image is None:
                        st.error("Failed to decode the image. Please try another file.")
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                        # Display original image
                        st.subheader("Original Image")
                        st.image(image, use_container_width=True)
                        
                        # Initialize variables to avoid reference errors
                        custom_results = None
                        lib_results = None
                        
                        # Process based on selected implementation
                        if implementation_type == "Custom Implementation" or implementation_type == "Comparison (Both)":
                            # Process with Custom Canny Pipeline
                            with st.spinner("Processing image with custom Canny edge detection..."):
                                custom_results = apply_canny_pipeline(image)
                            
                            # Display execution time
                            st.success(f"Custom processing completed in {custom_results['execution_time']*1000:.2f} ms")
                        
                        if implementation_type == "Library Implementation" or implementation_type == "Comparison (Both)":
                            # Process with Library Implementation
                            with st.spinner("Processing image with library Canny edge detection..."):
                                lib_results = apply_library_canny(image)
                            
                            # Display execution time
                            st.success(f"Library processing completed in {lib_results['execution_time']*1000:.2f} ms")
                        
                        # Display results based on implementation type
                        if implementation_type == "Custom Implementation":
                            # Display final result
                            st.subheader("Custom Canny Edge Detection Result")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(custom_results['hysteresis_img'], caption="After Hysteresis (Final Result)", use_container_width=True)
                            with col2:
                                st.image(custom_results['color_edges'], caption="Edge Direction Colormap", use_container_width=True)
                            
                            # Display all steps if selected
                            if show_all_steps:
                                st.subheader("Complete Custom Canny Edge Detection Pipeline")
                                
                                # Create a figure with subplots for visualization
                                fig, axes = plt.subplots(2, 6, figsize=(20, 10))
                                fig.suptitle('Custom Canny Edge Detection Pipeline Steps', fontsize=16)
                                
                                # First row
                                axes[0, 0].imshow(image)
                                axes[0, 0].set_title('Original')
                                axes[0, 0].axis('off')
                                
                                axes[0, 1].imshow(custom_results['gray_image'], cmap='gray')
                                axes[0, 1].set_title('Grayscale')
                                axes[0, 1].axis('off')
                                
                                axes[0, 2].imshow(custom_results['gaussian_image'], cmap='gray')
                                axes[0, 2].set_title(f'Gaussian (σ={sigma})')
                                axes[0, 2].axis('off')
                                
                                axes[0, 3].imshow(custom_results['fx'], cmap='gray')
                                axes[0, 3].set_title('Horizontal Edges (fx)')
                                axes[0, 3].axis('off')
                                
                                axes[0, 4].imshow(custom_results['fy'], cmap='gray')
                                axes[0, 4].set_title('Vertical Edges (fy)')
                                axes[0, 4].axis('off')
                                
                                axes[0, 5].imshow(custom_results['magnitude'], cmap='gray')
                                axes[0, 5].set_title('Edge Magnitude')
                                axes[0, 5].axis('off')
                                
                                # Second row
                                normalized_angle = ((custom_results['angle']) / 360)
                                axes[1, 0].imshow(normalized_angle, cmap='gray')
                                axes[1, 0].set_title('Angle')
                                axes[1, 0].axis('off')
                                
                                axes[1, 1].imshow(custom_results['color_edges'])
                                axes[1, 1].set_title('Colorized Edges')
                                axes[1, 1].axis('off')
                                
                                axes[1, 2].imshow(custom_results['non_max_img'], cmap='gray')
                                axes[1, 2].set_title('Non-max Suppression')
                                axes[1, 2].axis('off')
                                
                                axes[1, 3].imshow(custom_results['threshold_img'], cmap='gray')
                                axes[1, 3].set_title(f'Double Threshold\n(Hi:{thhi}, Lo:{thlo})')
                                axes[1, 3].axis('off')
                                
                                axes[1, 4].imshow(custom_results['hysteresis_img'], cmap='gray')
                                axes[1, 4].set_title('Hysteresis')
                                axes[1, 4].axis('off')
                                
                                # Make one plot for kernel visualization
                                gauss_kernel = gaussian_kernel(kernel_size, sigma)
                                axes[1, 5].imshow(gauss_kernel, cmap='viridis')
                                axes[1, 5].set_title(f'Gaussian Kernel\n(size={kernel_size}, σ={sigma})')
                                axes[1, 5].axis('off')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Performance metrics
                                st.subheader("Edge Detection Metrics")
                                metrics_data = {
                                    "Parameter": ["Execution Time", "Edge Pixels (%)", "Strong Edge Pixels (%)", "Weak Edge Pixels (%)"],
                                    "Value": [
                                        f"{custom_results['execution_time']*1000:.2f} ms",
                                        f"{np.count_nonzero(custom_results['hysteresis_img']) / custom_results['hysteresis_img'].size * 100:.2f}%",
                                        f"{np.count_nonzero(custom_results['threshold_img'] == 255) / custom_results['threshold_img'].size * 100:.2f}%",
                                        f"{np.count_nonzero(custom_results['threshold_img'] == 128) / custom_results['threshold_img'].size * 100:.2f}%"
                                    ]
                                }
                                st.table(metrics_data)
                                
                                # Color direction legend
                                st.subheader("Edge Direction Color Legend")
                                legend_data = {
                                    "Direction": ["0° (Horizontal)", "45° (Diagonal)", "90° (Vertical)", "135° (Diagonal)"],
                                    "Color": ["Yellow", "Green", "Blue", "Magenta"],
                                    "Meaning": ["East-West edges", "Northeast-Southwest edges", "North-South edges", "Northwest-Southeast edges"]
                                }
                                st.table(legend_data)

                        elif implementation_type == "Library Implementation":
                            # Display final result
                            st.subheader("Library Canny Edge Detection Result")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(lib_results['canny_edges'], caption=f"Library Canny (Low={lib_low}, High={lib_high})", use_container_width=True)
                            with col2:
                                st.image(lib_results['color_edges'], caption="Edge Direction Colormap", use_container_width=True)
                            
                            # Display all steps if selected
                            if show_all_steps:
                                st.subheader("Complete Library Canny Edge Detection Pipeline")
                                
                                # Create a figure with subplots for visualization
                                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                                fig.suptitle('Library Canny Edge Detection Pipeline Steps', fontsize=16)
                                
                                # First row
                                axes[0, 0].imshow(image)
                                axes[0, 0].set_title('Original')
                                axes[0, 0].axis('off')
                                
                                axes[0, 1].imshow(lib_results['gray_image'], cmap='gray')
                                axes[0, 1].set_title('Grayscale')
                                axes[0, 1].axis('off')
                                
                                axes[0, 2].imshow(lib_results['gaussian_image'], cmap='gray')
                                axes[0, 2].set_title(f'Gaussian (σ={lib_sigma})')
                                axes[0, 2].axis('off')
                                
                                # Second row
                                axes[1, 0].imshow(lib_results['magnitude'], cmap='gray')
                                axes[1, 0].set_title('Gradient Magnitude')
                                axes[1, 0].axis('off')
                                
                                axes[1, 1].imshow(lib_results['canny_edges'], cmap='gray')
                                axes[1, 1].set_title(f'Canny Edges\n(Low={lib_low}, High={lib_high})')
                                axes[1, 1].axis('off')
                                
                                axes[1, 2].imshow(lib_results['color_edges'])
                                axes[1, 2].set_title('Colorized Edges')
                                axes[1, 2].axis('off')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Performance metrics
                                st.subheader("Edge Detection Metrics")
                                metrics_data = {
                                "Parameter": ["Execution Time", "Edge Pixels (%)"],
                                "Value": [
                                    f"{lib_results['execution_time']*1000:.2f} ms",
                                    f"{np.count_nonzero(lib_results['canny_edges']) / lib_results['canny_edges'].size * 100:.2f}%"
                                ]
                            }
                            st.table(metrics_data)

                        else:  # Comparison mode
                            # Display comparison of results
                            st.subheader("Comparison: Custom vs Library Implementation")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.image(custom_results['hysteresis_img'], caption="Custom Implementation", use_container_width=True)
                            
                            with col2:
                                st.image(lib_results['canny_edges'], caption="Library Implementation", use_container_width=True)
                            
                            # Display execution time comparison
                            st.subheader("Performance Comparison")
                            performance_data = {
                                "Implementation": ["Custom", "Library"],
                                "Execution Time": [
                                    f"{custom_results['execution_time']*1000:.2f} ms",
                                    f"{lib_results['execution_time']*1000:.2f} ms"
                                ],
                                "Edge Pixels (%)": [
                                    f"{np.count_nonzero(custom_results['hysteresis_img']) / custom_results['hysteresis_img'].size * 100:.2f}%",
                                    f"{np.count_nonzero(lib_results['canny_edges']) / lib_results['canny_edges'].size * 100:.2f}%"
                                ]
                            }
                            st.table(performance_data)
                            
                            if show_all_steps:
                                # Side by side comparison of all steps
                                st.subheader("Side by Side Pipeline Comparison")
                                
                                # Create comparison tabs
                                compare_tabs = st.tabs(["Original", "Gaussian", "Gradients", "Edges"])
                                
                                with compare_tabs[0]:
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.image(image, caption="Original Image", use_container_width=True)
                                    with c2:
                                        st.image(custom_results['gray_image'], caption="Grayscale", use_container_width=True)
                                
                                with compare_tabs[1]:
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.image(custom_results['gaussian_image'], caption=f"Custom Gaussian (σ={sigma})", use_container_width=True)
                                    with c2:
                                        st.image(lib_results['gaussian_image'], caption=f"Library Gaussian (σ={lib_sigma})", use_container_width=True)
                                
                                with compare_tabs[2]:
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.image(custom_results['magnitude'], caption="Custom Gradient Magnitude", use_container_width=True)
                                    with c2:
                                        st.image(lib_results['magnitude'], caption="Library Gradient Magnitude", use_container_width=True)
                                
                                with compare_tabs[3]:
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.image(custom_results['color_edges'], caption="Custom Edge Directions", use_container_width=True)
                                    with c2:
                                        st.image(lib_results['color_edges'], caption="Library Edge Directions", use_container_width=True)
                                
                                # Technical comparison
                                st.subheader("Technical Parameters Comparison")
                                tech_data = {
                                    "Parameter": ["Gaussian Sigma", "Kernel Size", "Low Threshold", "High Threshold"],
                                    "Custom Implementation": [f"{sigma}", f"{kernel_size}x{kernel_size}", f"{thlo}", f"{thhi}"],
                                    "Library Implementation": [f"{lib_sigma}", f"{lib_aperture}x{lib_aperture}", f"{lib_low}", f"{lib_high}"]
                                }
                                st.table(tech_data)
                    
                except Exception as e:
                    st.error(f"Error processing the image: {str(e)}")
                    st.exception(e)
            else:
                st.info("Please upload an image in the sidebar to visualize Canny edge detection.")
    with tab2:
        st.subheader("Sharpening")
        # Advanced parameters
        st.write("Adjust parameters for Sharpening Image")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.slider("Sharpening Weight", 0.1, 2.0, 1.0, 0.1)
            method = st.selectbox("Sharpening Method", ["Laplacian", "Unsharp Masking"])
        
        with col2:
            if method == "Unsharp Masking":
                alpha = st.slider("Alpha (Blend Factor)", 0.1, 3.0, 0.7, 0.1)
                kernel_size = st.slider("Kernel Size", 1, 11, 3, 2)
                if kernel_size % 2 == 0:  # Ensure kernel size is odd
                    kernel_size += 1
                sigma = st.slider("Sigma", 0.1, 5.0, 0.5, 0.1)
                
        # Proses gambar dari uploader yang sudah ada
        uploaded_file = st.session_state.get("task2_uploader", None)
        
        # Define Laplacian operators
        HL = np.array([[0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0]])
        HL_x = np.array([1.0, -2.0, 1.0])
        HL_y = np.array([[1.0], [-2.0], [1.0]])
        
        # Function to apply Laplacian in X direction
        def laplacian_x(image):
            x = cv2.filter2D(image, -1, HL_x)
            x_abs = np.abs(x)
            return cv2.normalize(x_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        # Function to apply Laplacian in Y direction
        def laplacian_y(image):
            y = cv2.filter2D(image, -1, HL_y)
            y_abs = np.abs(y)
            return cv2.normalize(y_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        # Function to apply Laplacian in both X and Y directions
        def laplacianXY(image):
            xy = cv2.filter2D(image, -1, HL)
            xy_abs = np.abs(xy)
            return cv2.normalize(xy_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        # Function to convert to uint8
        def to_uint8(image):
            return np.clip(image, 0, 255).astype(np.uint8)
        
        # Function for unsharp masking
        def unsharp_mask(img, alpha, kernel_size, sigma):
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            mask = img - blurred
            unsharp = img + (alpha * mask)
            return to_uint8(unsharp)
        
        # Apply sharpening if an image is uploaded
        if uploaded_file is not None:
            # Baca file yang diupload
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Reset file pointer agar file bisa dibaca lagi jika perlu
            uploaded_file.seek(0)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
            
            # Measure execution time for different operations
            runtime_data = []
            
            if method == "Laplacian":
                # Laplacian - manual & library
                start_time = time.time()
                x_laplacian = laplacian_x(blur)
                x_laplacian_time = time.time() - start_time
                runtime_data.append(("X-Laplacian", x_laplacian_time))
                
                start_time = time.time()
                y_laplacian = laplacian_y(blur)
                y_laplacian_time = time.time() - start_time
                runtime_data.append(("Y-Laplacian", y_laplacian_time))
                
                start_time = time.time()
                xy_laplacian = x_laplacian + y_laplacian
                xy_laplacian_time = time.time() - start_time
                runtime_data.append(("XY-Laplacian (Manual)", xy_laplacian_time))
                
                start_time = time.time()
                xy_sharpening = to_uint8(gray_image - (weight * xy_laplacian))
                xy_sharpening_time = time.time() - start_time
                runtime_data.append(("XY-Sharpening", xy_sharpening_time))
                
                start_time = time.time()
                laplacian_hasil = laplacianXY(blur)
                laplacian_lib_time = time.time() - start_time
                runtime_data.append(("Laplacian (Library)", laplacian_lib_time))
                
                start_time = time.time()
                sharpening_hasil = to_uint8(gray_image - (weight * laplacian_hasil))
                sharpening_lib_time = time.time() - start_time
                runtime_data.append(("Sharpening (Library)", sharpening_lib_time))
                
                # Display Laplacian results
                st.subheader("Laplacian Filtering Results")
                fig, ax = plt.subplots(2, 3, figsize=(15, 10))
                
                ax[0, 0].imshow(x_laplacian, cmap='gray')
                ax[0, 0].set_title("X-Laplacian")
                ax[0, 0].axis('off')
                
                ax[0, 1].imshow(y_laplacian, cmap='gray')
                ax[0, 1].set_title("Y-Laplacian")
                ax[0, 1].axis('off')
                
                ax[0, 2].imshow(xy_laplacian, cmap='gray')
                ax[0, 2].set_title("XY-Laplacian Manual (X+Y)")
                ax[0, 2].axis('off')
                
                ax[1, 0].imshow(xy_sharpening, cmap='gray')
                ax[1, 0].set_title(f"XY-Sharpening (w={weight:.1f})")
                ax[1, 0].axis('off')
                
                ax[1, 1].imshow(laplacian_hasil, cmap='gray')
                ax[1, 1].set_title("Laplacian Filter")
                ax[1, 1].axis('off')
                
                ax[1, 2].imshow(sharpening_hasil, cmap='gray')
                ax[1, 2].set_title(f"Sharpening Filter (w={weight:.1f})")
                ax[1, 2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Allow user to download the sharpened image
                st.download_button(
                    label="Download Sharpened Image",
                    data=cv2.imencode('.png', sharpening_hasil)[1].tobytes(),
                    file_name="sharpened_image.png",
                    mime="image/png"
                )
            
            else:  # Unsharp Masking
                start_time = time.time()
                hasil_unsharp = unsharp_mask(gray_image, alpha, kernel_size, sigma)
                unsharp_time = time.time() - start_time
                runtime_data.append(("Unsharp Masking", unsharp_time))
                
                # Display Unsharp Masking results
                st.subheader("Unsharp Masking Results")
                fig, ax = plt.subplots(1, 2, figsize=(15, 7))
                
                ax[0].imshow(gray_image, cmap='gray')
                ax[0].set_title("Original Gray Image")
                ax[0].axis('off')
                
                ax[1].imshow(hasil_unsharp, cmap='gray')
                ax[1].set_title(f"Unsharp Masking (α={alpha:.1f}, σ={sigma:.1f}, kernel={kernel_size})")
                ax[1].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Allow user to download the sharpened image
                st.download_button(
                    label="Download Sharpened Image",
                    data=cv2.imencode('.png', hasil_unsharp)[1].tobytes(),
                    file_name="unsharp_masked_image.png",
                    mime="image/png"
                )
            
            # Runtime Analysis
            st.subheader("Runtime Analysis")
            
            # Create runtime comparison bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            operations = [item[0] for item in runtime_data]
            times = [item[1] * 1000 for item in runtime_data]  # Convert to milliseconds
            
            bars = ax.bar(operations, times, color='skyblue')
            ax.set_ylabel('Execution Time (ms)')
            ax.set_title('Performance Comparison of Sharpening Operations')
            ax.set_xticklabels(operations, rotation=45, ha='right')
            
            # Add labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f} ms',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display runtime data in a table
            runtime_df = pd.DataFrame(runtime_data, columns=['Operation', 'Time (s)'])
            runtime_df['Time (ms)'] = runtime_df['Time (s)'] * 1000
            runtime_df = runtime_df.drop(columns=['Time (s)'])
            st.dataframe(runtime_df)
            
        else:
            st.info("Please upload an image in the Home tab first.")

elif task_choice == "Task 3: Corner, Line, and Circle Detection":
    st.header("Task 3: Corner, Line, and Circle Detection")
    
    # Create tabs for different detection methods
    tab1, tab2, tab3 = st.tabs(["Corner Detection", "Line Detection", "Circle Detection"])           
    with tab1:
        st.subheader("Corner Detection")
        st.write("Adjust parameters for Corner Detection")
        uploaded_corner = st.file_uploader("Upload an image for advanced analysis", type=["jpg", "jpeg", "png"], key="task3_uploader")

        # Parameters for corner detection
        a = st.slider("a Value", min_value=0.01, max_value=0.5, value=0.04, step=0.01)
        threshold = st.slider("Threshold Value", min_value=100, max_value=10000000, value=101000, step=1000)
       
        if uploaded_corner is not None:
            file_bytes = np.asarray(bytearray(uploaded_corner.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           
            def harris_manual(gray, a=0.5, threshold=1e5):
                Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                
                Ixx = Ix ** 2
                Iyy = Iy ** 2
                Ixy = Ix * Iy
                
                A = cv2.GaussianBlur(Ixx, (3,3), sigmaX=1)
                B = cv2.GaussianBlur(Iyy, (3,3), sigmaX=1)
                C = cv2.GaussianBlur(Ixy, (3,3), sigmaX=1)
                
                detM = A * B - C ** 2
                trace = A + B
                
                Q = detM - a * (trace ** 2)
                
                corners = np.zeros_like(gray)
                corners[Q > threshold] = 255
                
                img_corners = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                img_corners[corners == 255] = [255, 0, 0]
                
                return img_corners
        
            def harris_library(gray, a=0.04, threshold=1e5):
                gray_float = np.float32(gray)
                dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=a)
                dst = cv2.dilate(dst, None)
                
                img_harris = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                img_harris[dst > threshold*dst.max()] = [0, 255, 0]  # Fixed threshold application
                
                return img_harris
                
            start_manual = time.time()
            result_manual = harris_manual(gray, a=a, threshold=threshold)
            end_manual = time.time()
            
            start_lib = time.time()
            result_lib = harris_library(gray, a=a, threshold=0.01)  # Using relative threshold
            end_lib = time.time()
            
            time_manual = end_manual - start_manual
            time_lib = end_lib - start_lib

            st.write("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Display manual result
            st.write(f"Manual Harris Corner Detection (Red) - Time: {time_manual:.4f} seconds")
            st.image(cv2.cvtColor(result_manual, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Display library result
            st.write(f"Library Harris Corner Detection (Green) - Time: {time_lib:.4f} seconds")
            st.image(cv2.cvtColor(result_lib, cv2.COLOR_BGR2RGB), channels="RGB")

    with tab2:
        st.subheader("Line Detection")
        
        # Separate file uploader for line detection
        uploaded_file_line = st.file_uploader("Upload an image for line detection", type=["jpg", "jpeg", "png"], key="line_uploader")
        
        if uploaded_file_line is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file_line.read()), dtype=np.uint8)
            line_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gray_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
            
            # Parameters for line detection
            theta_steps = st.slider("Theta Steps", min_value=90, max_value=500, step=10, value=200)
            threshold = st.slider("Threshold", min_value=10, max_value=1000, step=10, value=200)
            
            # Define Hough line detection functions
            def hough_line_detection_theoretical(gray_image, theta_steps=200, threshold=200):
                edges = cv2.Canny(gray_image, 50, 150)

                H, W = edges.shape
                x_r, y_r = W // 2, H // 2

                m = theta_steps
                d_theta = np.pi / m
                theta_vals = np.linspace(0, np.pi, m)

                r_max = int(np.hypot(W, H))
                d_r = 1
                n_r = 2 * r_max
                j_0 = r_max

                accumulator = np.zeros((n_r, m), dtype=np.uint8)

                ys, xs = np.nonzero(edges)
                for u, v in zip(xs, ys):
                    for k, theta_k in enumerate(theta_vals):
                        r_k = (u - x_r) * np.cos(theta_k) + (v - y_r) * np.sin(theta_k)
                        j = int(round(r_k / d_r)) + j_0
                        if 0 <= j < n_r:
                            accumulator[j, k] += 1

                lines = np.argwhere(accumulator > threshold)
                result_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

                for j, k in lines:
                    theta_k = theta_vals[k]
                    r_k = (j - j_0) * d_r

                    a = np.cos(theta_k)
                    b = np.sin(theta_k)
                    x0 = a * r_k + x_r
                    y0 = b * r_k + y_r
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                return result_image, accumulator, edges, len(lines)

            def hough_line_library(gray_image, threshold=200):
                edges = cv2.Canny(gray_image, 50, 150)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

                result_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                num_lines = 0

                if lines is not None:
                    num_lines = len(lines)
                    for rho_theta in lines:
                        rho, theta = rho_theta[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

                return result_image, edges, num_lines
            
            # Process and display results
            start_manual = time.time()
            result_manual, accumulator, edges_manual, manual_lines = hough_line_detection_theoretical(gray_image, theta_steps, threshold)
            end_manual = time.time()

            start_lib = time.time()
            result_lib, edges_lib, lib_lines = hough_line_library(gray_image, threshold)
            end_lib = time.time()

            time_manual = end_manual - start_manual
            time_lib = end_lib - start_lib
            
            # Display original image
            st.write("Original Image")
            st.image(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Create columns for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Manual Implementation")
                
                st.write("Grayscale")
                st.image(gray_image, channels="GRAY")
                
                st.write("Edges")
                st.image(edges_manual, channels="GRAY")
                
                st.write("Accumulator")
                fig, ax = plt.subplots()
                ax.imshow(accumulator, cmap='hot', aspect='auto')
                ax.set_title('Accumulator (Manual)')
                ax.set_xlabel('Theta')
                ax.set_ylabel('Rho')
                st.pyplot(fig)
                
                st.write(f"Line Detection Results (Green) - {manual_lines} lines detected")
                st.write(f"Processing Time: {time_manual:.4f} seconds")
                st.image(cv2.cvtColor(result_manual, cv2.COLOR_BGR2RGB), channels="RGB")
            
            with col2:
                st.write("Library Implementation")
                
                st.write("Grayscale")
                st.image(gray_image, channels="GRAY")
                
                st.write("Edges")
                st.image(edges_lib, channels="GRAY")
                
                st.write("No Accumulator (Library)")
                
                st.write(f"Line Detection Results (Blue) - {lib_lines} lines detected")
                st.write(f"Processing Time: {time_lib:.4f} seconds")
                st.image(cv2.cvtColor(result_lib, cv2.COLOR_BGR2RGB), channels="RGB")
                
        else:
            st.write("Please upload an image for line detection.")
        
    with tab3:
        st.subheader("Circle Detection")
        
        # Separate file uploader for circle detection
        uploaded_file_circle = st.file_uploader("Upload an image for circle detection", type=["jpg", "jpeg", "png"], key="circle_uploader")
        
        if uploaded_file_circle is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file_circle.read()), dtype=np.uint8)
            circle_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gray_circle = cv2.cvtColor(circle_image, cv2.COLOR_BGR2GRAY)
            
            # Add edge detection for the manual Hough transform
            blur = cv2.GaussianBlur(gray_circle, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            
            # Parameters for circle detection
            radius = st.slider("Circle Radius", min_value=0, max_value=100, step=10, value=25)
            threshold = st.slider("Threshold", min_value=0, max_value=200, step=10, value=150)
            
            # Define Hough circle detection functions
            def hough_circle_accumulator(edges, radius):
                height, width = edges.shape
                accumulator = np.zeros((height, width), dtype=np.uint64)
                ys, xs = np.nonzero(edges)
            
                for x, y in zip(xs, ys):
                    for theta in range(0, 360):
                        t = np.deg2rad(theta)
                        a = int(x - radius * np.cos(t))
                        b = int(y - radius * np.sin(t))
                        if 0 <= a < width and 0 <= b < height:
                            accumulator[b, a] += 1
            
                return accumulator
            
            def detect_circles(accumulator, threshold, radius):
                centers = np.argwhere(accumulator > threshold)
                circles = [(x, y, radius) for y, x in centers]
                return circles
            
            def draw_circles(image, circles, color=(0, 255, 0)):
                output = np.stack([image]*3, axis=-1) if len(image.shape) == 2 else image.copy()
                for x, y, r in circles:
                    # Pertebal lingkaran menjadi 2 piksel
                    cv2.circle(output, (x, y), r, color, 2)
                    # Tambahkan titik tengah lingkaran
                    cv2.circle(output, (x, y), 3, (0, 0, 255), -1)  # Titik berwarna merah
                return output
            
            def hough_circle_library(gray, minRadius, maxRadius, param1=100, param2=30):
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                           param1=param1, param2=param2,
                                           minRadius=minRadius, maxRadius=maxRadius)
                result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for x, y, r in circles[0]:
                        # Pertebal lingkaran menjadi 2 piksel
                        cv2.circle(result, (x, y), r, (255, 0, 0), 2)
                        # Tambahkan titik tengah lingkaran
                        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)  # Titik berwarna merah
                    return result, circles[0]
                return result, []
                
            # Manual Hough transform
            start_manual = time.time()
            accumulator = hough_circle_accumulator(edges, radius)
            manual_circles = detect_circles(accumulator, threshold, radius)
            result_manual = draw_circles(gray_circle, manual_circles, color=(0, 255, 0))
            end_manual = time.time()
            time_manual = end_manual - start_manual
            
            # OpenCV built-in Hough Circle detection
            start_lib = time.time()
            result_lib, circles_lib = hough_circle_library(gray_circle, minRadius=radius-5, maxRadius=radius+5)
            end_lib = time.time()
            time_lib = end_lib - start_lib
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Manual Hough Transform")
                st.image(result_manual, channels="BGR", use_container_width=True)
                st.write(f"Execution time: {time_manual:.4f} seconds")
                st.write(f"Circles detected: {len(manual_circles)}")
                
            with col2:
                st.subheader("OpenCV Hough Circles")
                st.image(result_lib, channels="BGR", use_container_width=True)
                st.write(f"Execution time: {time_lib:.4f} seconds")
                st.write(f"Circles detected: {len(circles_lib)}")
            
            # Create detailed visualization with matplotlib
            st.subheader("Detailed Visualization")
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            # First row
            axes[0, 0].imshow(gray_circle, cmap='gray')
            axes[0, 0].set_title('Gray Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(edges, cmap='gray')
            axes[0, 1].set_title('Edges (Canny)')
            axes[0, 1].axis('off')
            
            im = axes[0, 2].imshow(accumulator, cmap='hot')
            axes[0, 2].set_title('Accumulator (Manual)')
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
            
            # Second row
            axes[1, 0].imshow(cv2.cvtColor(result_manual, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f'Manual Circles: {len(manual_circles)} \nTime: {time_manual:.4f} sec')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(cv2.cvtColor(result_lib, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title(f'Library Circles: {len(circles_lib)} \nTime: {time_lib:.4f} sec')
            axes[1, 1].axis('off')
            
            # Keep the last plot empty or use for additional metrics
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display circle details
            st.subheader("Circle Detection Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Manual Circle Details")
                st.write(f"Accumulator max value: {accumulator.max()}")
                st.write(f"Number of circles: {len(manual_circles)}")
                if len(manual_circles) > 0:
                    circle_data = []
                    for i, (x, y, r) in enumerate(manual_circles):
                        circle_data.append({"Circle": i+1, "Center X": x, "Center Y": y, "Radius": r})
                    st.table(circle_data)
            
            with col2:
                st.write("### OpenCV Circle Details")
                st.write(f"Number of circles: {len(circles_lib)}")
                if len(circles_lib) > 0:
                    circle_data = []
                    for i, (x, y, r) in enumerate(circles_lib):
                        circle_data.append({"Circle": i+1, "Center X": int(x), "Center Y": int(y), "Radius": int(r)})
                    st.table(circle_data)

    # Display message when no image is uploaded
    for tab in [tab1, tab2, tab3]:
        with tab:
            st.write("Please upload an image to perform detection.")


            
elif task_choice == "Final Project":
    st.header("🔬 FISH and DISH Analyzer")
    st.markdown("---")
    st.subheader("📁 Upload Files")
        
    def load_uploaded_image(uploaded_file):
        if uploaded_file is not None:
            # Convert uploaded file to opencv format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        
        return None   
    fish_tab, dish_tab = st.tabs(["🔬 FISH Analysis", "🎨 DISH Analysis"])
    
    def plot_histogram_with_image(image, title="Histogram", colormap="gray"):
        col1, col2 = st.columns(2)
        if image.dtype != np.uint8:
            image_disp = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        else:
            image_disp = image

        # Histogram
        hist = ndi.histogram(image_disp, min=0, max=255, bins=256)
        # Plot histogram
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(hist, color='black')
        ax.set_title(title)
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Number of Pixels")
        ax.grid(True)
        col1.pyplot(fig)
        # Gambar sumber
        col2.image(image_disp, caption="Source Image", use_column_width=True, clamp=True, channels="GRAY")

    def plot_rgb_histogram(image, key_prefix=""):
        if st.checkbox("Tampilkan Histogram RGB per Channel", key=f"{key_prefix}_rgbhist"):
            if image.dtype != np.float32 and image.dtype != np.float64:
                image = image / 255.0

            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]

            hist_r, bins_r = exposure.histogram(r)
            hist_g, bins_g = exposure.histogram(g)
            hist_b, bins_b = exposure.histogram(b)

            fig, axs = plt.subplots(2, 3, figsize=(15, 8))

            axs[0, 0].plot(bins_r, hist_r, color='red')
            axs[0, 1].plot(bins_g, hist_g, color='green')
            axs[0, 2].plot(bins_b, hist_b, color='blue')

            axs[1, 0].imshow(r, cmap='gray')
            axs[1, 1].imshow(g, cmap='gray')
            axs[1, 2].imshow(b, cmap='gray')

            for i in range(3):
                axs[0, i].set_title(f'Histogram {"RGB"[i]} Channel')
                axs[1, i].set_title(f'{"RGB"[i]} Channel Image')
                axs[0, i].grid(True)
                axs[1, i].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

            return r, g, b
        else:
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
            return r, g, b


    def apply_clahe(image_channel, label="CLAHE Result", key_prefix=""):
        value_cl = st.slider(
            "Choose Clip Limit value",
            min_value=0.001,
            max_value=0.02,
            value=0.007,
            step=0.001,
            key=f"{key_prefix}_clip_limit"
        )

        im_clahe = exposure.equalize_adapthist(image_channel, clip_limit=value_cl)

        hist_clahe, bins_clahe = exposure.histogram(im_clahe)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(im_clahe, cmap='gray')
        axs[0].set_title(f'{label}')
        axs[0].axis('off')
        axs[1].plot(bins_clahe, hist_clahe, color='black')
        axs[1].set_title('Histogram after CLAHE')
        axs[1].set_xlabel('Intensity')
        axs[1].set_ylabel('Pixel Count')
        axs[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig)
        return im_clahe

    def apply_otsu_threshold(image, title="Otsu Thresholding", is_dish=False):
        """
        image       : channel citra grayscale (0-1 atau 0-255)
        title       : judul plot
        is_dish     : True jika jenis citra DISH (background putih)
        """
        if image.max() <= 1.0:
            image_uint = (image * 255).astype(np.uint8)
        else:
            image_uint = image.astype(np.uint8)

        thresh_val = filters.threshold_otsu(image_uint)
        if is_dish:
            binary_mask = image_uint < thresh_val  # background terang → ambil gelap
        else:
            binary_mask = image_uint > thresh_val  # background gelap → ambil terang

        # Plot hasil
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))

        axs[0].imshow(image_uint, cmap='Blues')
        axs[0].set_title('Original Channel (uint8)')
        axs[0].axis('off')

        axs[1].hist(image_uint.ravel(), bins=256, color='blue')
        axs[1].axvline(thresh_val, color='red', linestyle='--')
        axs[1].set_title(f'Histogram + Otsu Threshold = {thresh_val:.1f}')

        axs[2].imshow(binary_mask, cmap='gray')
        axs[2].set_title('Thresholded Mask\n(Foreground = White)')
        axs[2].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

        return binary_mask

    def filter_and_label_cells(binary_mask, key_prefix="fish"):
        st.markdown("### Parameter Filtering dan Morphology")

        min_obj_size = st.number_input(
            "Minimum Object Size (remove_small_objects)",
            min_value=100,
            max_value=20000,
            value=8000 if key_prefix == "fish" else 8000,
            step=100,
            key=f"{key_prefix}_min_obj"
        )
        min_hole_size = st.number_input(
            "Minimum Hole Size (fill_holes)",
            min_value=100,
            max_value=10000,
            value=1200 if key_prefix == "fish" else 2500,
            step=100,
            key=f"{key_prefix}_min_hole"
        )
        smooth_radius = st.slider(
            "Smooth Kontur (Closing Radius)",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            key=f"{key_prefix}_smooth"
        )

        mask_no_small = morphology.remove_small_objects(binary_mask, min_size=min_obj_size)

        mask_filled = np.logical_not(
            morphology.remove_small_objects(np.logical_not(mask_no_small), min_size=min_hole_size))

        if smooth_radius > 0:
            mask_smoothed = morphology.binary_closing(mask_filled, morphology.disk(smooth_radius))
        else:
            mask_smoothed = mask_filled.copy()

        labels, nlabels = ndi.label(mask_smoothed)
        rand_cmap = ListedColormap(np.random.rand(256, 3))
        labels_for_display = np.where(labels > 0, labels, np.nan)

        # Plot
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
        axs[0].imshow(binary_mask, cmap='gray')
        axs[0].set_title('Mask Awal (Threshold)')
        axs[0].axis('off')
        axs[1].imshow(mask_no_small, cmap='gray')
        axs[1].set_title('Remove Small Object')
        axs[1].axis('off')
        axs[2].imshow(mask_filled, cmap='gray')
        axs[2].set_title('Fill Holes')
        axs[2].axis('off')
        axs[3].imshow(mask_smoothed, cmap='gray')
        axs[3].set_title('Smoothed (Closing)')
        axs[3].axis('off')
        axs[4].imshow(labels_for_display, cmap=rand_cmap)
        axs[4].set_title(f'Labeled = {nlabels}')
        axs[4].axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        st.success(f"✅ Total objek terdeteksi setelah filtering: **{nlabels} objek**")
        return mask_smoothed, labels, nlabels

    def watershed_segmentation(image_segmented, image, key_prefix="fish", show_plot=True):
        st.markdown("### Parameter Watershed")

        sigma_default = 1.5 if key_prefix == "fish" else 1.8
        min_dist_default = 30 if key_prefix == "fish" else 60
        footprint_default = 20 if key_prefix == "fish" else 21

        sigma = st.slider(
            "Gaussian Sigma (Smooth Distance Map)",
            min_value=0.0, max_value=5.0, value=sigma_default, step=0.1,
            key=f"{key_prefix}_sigma"
        )
        min_distance = st.slider(
            "Minimum Distance antar Seed", min_value=1, max_value=50,
            value=min_dist_default, step=1,
            key=f"{key_prefix}_min_distance"
        )
        footprint_size = st.slider(
            "Ukuran Footprint Seed", min_value=1, max_value=30,
            value=footprint_default, step=1,
            key=f"{key_prefix}_footprint"
        )

        distance = ndi.distance_transform_edt(image_segmented)
        distance_smooth = gaussian(distance, sigma=sigma)
        coordinates = feature.peak_local_max(
            distance_smooth,
            labels=image_segmented,
            min_distance=min_distance,
            footprint=np.ones((footprint_size, footprint_size))
        )
        local_maxi = np.zeros_like(distance, dtype=bool)
        local_maxi[tuple(coordinates.T)] = True
   
        markers = measure.label(local_maxi)

        labels_ws = segmentation.watershed(-distance, markers, mask=image_segmented)

        boundaries = find_boundaries(labels_ws, mode='inner')
        thick_boundaries = binary_dilation(boundaries, disk(1))

        image_with_lines = image.copy()
        if image_with_lines.dtype != np.uint8:
            image_with_lines = (image_with_lines * 255).clip(0, 255).astype(np.uint8)
        image_with_lines[thick_boundaries] = [255, 255, 0]

        nlabels = len(np.unique(labels_ws)) - 1  # Kurangi background
        st.success(f"✅ Total objek terdeteksi setelah watershed: {nlabels}")
        if show_plot:
            fig, axs = plt.subplots(1, 4, figsize=(18, 6))
            axs[0].imshow(distance, cmap='magma')
            axs[0].set_title('Distance Transform')
            axs[0].axis('off')

            axs[1].imshow(color.label2rgb(labels_ws, image=image, bg_label=0))
            axs[1].set_title("Watershed Result (Sel Terpisah)")
            axs[1].axis('off')

            axs[2].imshow(thick_boundaries, cmap='gray')
            axs[2].set_title("Boundary Lines (Binary)")
            axs[2].axis('off')

            axs[3].imshow(image_with_lines)
            axs[3].set_title("Overlay: Watershed Line on Original Image")
            axs[3].axis('off')

            plt.tight_layout()
            st.pyplot(fig)

        return labels_ws, thick_boundaries, image_with_lines

    def extract_gt_mask_from_yellow(gt_rgb):
        r = gt_rgb[:, :, 0]
        g = gt_rgb[:, :, 1]
        b = gt_rgb[:, :, 2]

        yellow_mask = (r > 200) & (g > 200) & (b < 100)
        yellow_mask_dilated = binary_dilation(yellow_mask, disk(1))
        closed = binary_closing(yellow_mask_dilated, disk(3))
        gt_mask = ndi.binary_fill_holes(closed)
        return gt_mask

    def evaluate_segmentation_result(gt_array, segmented_mask):
        try:
            if len(gt_array.shape) == 3:
                gt_binary = cv2.cvtColor(gt_array, cv2.COLOR_RGB2GRAY)
            else:
                gt_binary = gt_array.copy()
            
            _, gt_binary = cv2.threshold(gt_binary, 127, 255, cv2.THRESH_BINARY)
            gt_binary = gt_binary > 0
            
            if segmented_mask.dtype != bool:
                segmented_mask = segmented_mask > 0
            
            if gt_binary.shape != segmented_mask.shape:
                gt_binary = cv2.resize(gt_binary.astype(np.uint8), 
                                    (segmented_mask.shape[1], segmented_mask.shape[0]))
                gt_binary = gt_binary > 0
            
            intersection = np.logical_and(gt_binary, segmented_mask)
            union = np.logical_or(gt_binary, segmented_mask)
            
            # IoU (Intersection over Union)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            
            # Precision and Recall
            precision = np.sum(intersection) / np.sum(segmented_mask) if np.sum(segmented_mask) > 0 else 0
            recall = np.sum(intersection) / np.sum(gt_binary) if np.sum(gt_binary) > 0 else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Accuracy
            total_pixels = gt_binary.size
            correct_pixels = np.sum(gt_binary == segmented_mask)
            accuracy = correct_pixels / total_pixels
            
            # Display results in a nice format
            st.markdown("### 📊 Evaluation Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("IoU Score", f"{iou:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1 Score", f"{f1:.4f}")
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col6:
                st.metric("Dice Score", f"{f1:.4f}")  
                
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(gt_binary, cmap='gray')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            axes[1].imshow(segmented_mask, cmap='gray')
            axes[1].set_title('Segmentation Result')
            axes[1].axis('off')
            
            overlay = np.zeros((gt_binary.shape[0], gt_binary.shape[1], 3))
            overlay[:, :, 0] = gt_binary.astype(float)  # Red for ground truth
            overlay[:, :, 1] = segmented_mask.astype(float)  # Green for segmented
            overlay[:, :, 2] = intersection.astype(float)  # Blue for intersection
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay\n(Red:GT, Green:Seg, Blue:Intersect)')
            axes[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("### 📋 Results Interpretation")
            if iou > 0.7:
                st.success("🎉 Excellent segmentation quality!")
            elif iou > 0.5:
                st.info("👍 Good segmentation quality")
            elif iou > 0.3:
                st.warning("⚠️ Moderate segmentation quality")
            else:
                st.error("❌ Poor segmentation quality")
                
            with st.expander("📈 Detailed Metrics Explanation"):
                st.markdown(f"""
                **IoU (Intersection over Union)**: {iou:.4f}
                - Measures overlap between predicted and ground truth
                - Range: 0-1, higher is better
                
                **Precision**: {precision:.4f}
                - Of all pixels predicted as foreground, how many are correct?
                - Range: 0-1, higher is better
                
                **Recall (Sensitivity)**: {recall:.4f}
                - Of all actual foreground pixels, how many were detected?
                - Range: 0-1, higher is better
                
                **F1 Score**: {f1:.4f}
                - Harmonic mean of precision and recall
                - Range: 0-1, higher is better
                
                **Accuracy**: {accuracy:.4f}
                - Overall pixel-wise accuracy
                - Range: 0-1, higher is better
                """)
            
            return {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
            
        except Exception as e:
            st.error(f"Error in evaluation: {str(e)}")
            st.info("Please check if the ground truth image format is correct")
            return None

    def plot_chan_signal (im):
        if im.dtype != np.float32 and im.dtype != np.float64:
            image_rgb = im / 255.0
        else:
            image_rgb = im.copy()

        # Ekstrak channel merah (HER2) dan hijau (CEN17)
        her2_channel = image_rgb[:, :, 0]
        cen17_channel = image_rgb[:, :, 1]
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(her2_channel, cmap='Reds')
        axs[0].set_title("HER2 Channel (Red)")
        axs[0].axis('off')

        axs[1].imshow(cen17_channel, cmap='Greens')
        axs[1].set_title("CEN17 Channel (Green)")
        axs[1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        return her2_channel, cen17_channel

    def extract_and_plot_dish_signal_channels(im, cen17_mask_thresh=0.1):
      
        image_rgb = im / 255.0 if im.dtype != np.float32 and im.dtype != np.float64 else im.copy()

        cen17_channel = np.clip(image_rgb[:, :, 0] - image_rgb[:, :, 1], 0, 1)

        grayscale = rgb2gray(image_rgb)
      
        mask = cen17_channel > cen17_mask_thresh
        her2_channel = np.copy(grayscale)
        her2_channel[mask] = 1.0 
        her2_channel = 1.0 - her2_channel
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(cen17_channel, cmap='Reds')
        axs[0].set_title("DISH CEN17 Channel (Red - Green)")
        axs[0].axis('off')
        axs[1].imshow(her2_channel, cmap='gray')
        axs[1].set_title("DISH HER2 Channel (Grayscale - CEN17 masked)")
        axs[1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        return her2_channel, cen17_channel

    def apply_clahe_sig(image_channel, label="Channel", key_prefix=""):
        st.write(f"CLAHE diterapkan pada: **{label}**")

        clip_limit = st.number_input(
            "Clip Limit (manual input)",
            min_value=0.001,
            max_value=0.05,
            value=0.007,
            step=0.001,
            format="%.3f",
            key=f"{key_prefix}_clip_limit"
        )
        im_clahe = exposure.equalize_adapthist(
            image_channel,
            clip_limit=clip_limit,
            kernel_size=(32, 32)
        )
        hist_clahe, bins_clahe = exposure.histogram(im_clahe)
        # Pilih cmap dan warna histogram
        label_lower = label.lower()
        if "her2" in label_lower:
            cmap = "Reds"
            line_color = "red"
        elif "cen17" in label_lower:
            cmap = "Greens"
            line_color = "green"
        else:
            cmap = "gray"
            line_color = "black"

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(im_clahe, cmap=cmap)
        axs[0].set_title(f'CLAHE Result ({label})')
        axs[0].axis('off')
        axs[1].plot(bins_clahe, hist_clahe, color=line_color)
        axs[1].set_title(f'Histogram after CLAHE ({label})')
        axs[1].set_xlabel('Intensity')
        axs[1].set_ylabel('Pixel Count')
        axs[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig)
        return im_clahe

    def stretch_channel(channel, label="Channel", cmap="gray", key_prefix=""):
        st.subheader(f"{label} – Intensity Stretching")
        stretch_min = st.number_input(
            "Stretch range (manual input)",
            min_value=0.2,
            max_value=0.7,
            value=0.33,
            step=0.01,
            format="%.2f",
            key=f"{key_prefix}_stretch_min"
        )
        stretched = rescale_intensity(channel, in_range=(stretch_min, 1.0))

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(channel, cmap=cmap)
        axs[0].set_title(f"{label} Original")
        axs[0].axis('off')

        axs[1].imshow(stretched, cmap=cmap)
        axs[1].set_title(f"{label} Stretched (min={stretch_min})")
        axs[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        return stretched

    def detect_and_plot_signal_coords(image_rgb, her2_stretched, cen17_stretched,
                                    default_thresh_min=0.33, thresh_max=1.0,
                                    key_prefix="fish"):
        """
        Mendeteksi dan memvisualisasikan koordinat titik sinyal HER2 dan CEN17 dari hasil stretching.
        """
        col1, col2 = st.columns(2)
        with col1:
            her2_min = st.number_input(
                f"Threshold Minimum HER2 ({key_prefix})", 
                min_value=0.0, max_value=1.0,
                value=default_thresh_min, step=0.01, format="%.2f",
                key=f"{key_prefix}_her2_thresh"
            )
        with col2:
            cen17_min = st.number_input(
                f"Threshold Minimum CEN17 ({key_prefix})", 
                min_value=0.0, max_value=1.0,
                value=default_thresh_min, step=0.01, format="%.2f",
                key=f"{key_prefix}_cen17_thresh"
            )

        her2_coords = np.argwhere((her2_stretched >= her2_min) & (her2_stretched <= thresh_max))
        cen17_coords = np.argwhere((cen17_stretched >= cen17_min) & (cen17_stretched <= thresh_max))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_rgb)

        if her2_coords.size > 0:
            ax.scatter(
                her2_coords[:, 1], her2_coords[:, 0],
                s=5, facecolors='none', edgecolors='red', linewidths=0.5, label='HER2'
            )
        if cen17_coords.size > 0:
            ax.scatter(
                cen17_coords[:, 1], cen17_coords[:, 0],
                s=5, facecolors='none', edgecolors='lime', linewidths=0.5, label='CEN17'
            )

        ax.set_title("Titik Sinyal Terdeteksi Berdasarkan Threshold Manual")
        ax.axis('off')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc='upper right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        return her2_coords, cen17_coords

    def analyze_her2_cen17_ratio(labels_ws, her2_coords, cen17_coords):
    
        label_map = labels_ws
        regions = regionprops(label_map)

        her2_map = np.zeros_like(label_map, dtype=int)
        cen17_map = np.zeros_like(label_map, dtype=int)

        her2_map[her2_coords[:, 0], her2_coords[:, 1]] = 1
        cen17_map[cen17_coords[:, 0], cen17_coords[:, 1]] = 1

        region_ids, her2_counts, cen17_counts, ratios, statuses = [], [], [], [], []

        for region in regions:
            region_id = region.label
            coords = region.coords

            h_count = her2_map[coords[:, 0], coords[:, 1]].sum()
            c_count = cen17_map[coords[:, 0], coords[:, 1]].sum()
            ratio = h_count / c_count if c_count > 0 else 0

            if ratio > 2.0:
                status = "HER2-Positive"
            elif 1.8 <= ratio <= 2.0:
                status = "Equivocal"
            else:
                status = "HER2-Negative"

            region_ids.append(region_id)
            her2_counts.append(h_count)
            cen17_counts.append(c_count)
            ratios.append(ratio)
            statuses.append(status)

        df_ratio = pd.DataFrame({
            "Region": region_ids,
            "HER2_Count": her2_counts,
            "CEN17_Count": cen17_counts,
            "HER2/CEN17_Ratio": ratios,
            "HER2_Status": statuses
        })

        st.dataframe(df_ratio.style.format({'HER2/CEN17_Ratio': '{:.2f}'}), height=400)

        st.markdown("**Region HER2-Positive**")
        st.dataframe(df_ratio[df_ratio['HER2_Status'] == 'HER2-Positive'])
        return df_ratio

    def visualize_her2_cen17_classification(im, labels_ws, her2_coords, cen17_coords, df_ratio):

        boundaries = find_boundaries(labels_ws, mode='inner')
        overlay_img = label2rgb(labels_ws, image=im, bg_label=0)
        regions = measure.regionprops(labels_ws)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(im)  
        ax.imshow(boundaries, cmap='gray', alpha=0.5)  
        if her2_coords.shape[0] > 0:
            ax.scatter(
                her2_coords[:, 1], her2_coords[:, 0],
                s=5, facecolors='none', edgecolors='red', linewidths=0.5, label='HER2')

        if cen17_coords.shape[0] > 0:
            ax.scatter(
                cen17_coords[:, 1], cen17_coords[:, 0],
                s=5, facecolors='none', edgecolors='lime', linewidths=0.5, label='CEN17')

        # Bounding box + label per sel
        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            cy, cx = region.centroid

            status = df_ratio.loc[df_ratio["Region"] == region.label, "HER2_Status"].values[0]
            color_box = {
                'HER2-Positive': 'red',
                'Equivocal': 'yellow',
                'HER2-Negative': 'green'}[status]

            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor=color_box, linewidth=1.2)
            ax.add_patch(rect)
            ax.plot(cx, cy, 'o', color=color_box, markersize=4)
            ax.text(cx, cy, f'{region.label}', color='white', fontsize=6, ha='center', va='center')

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)

        ax.set_title("Visualisasi Status HER2/CEN17 + Titik Sinyal")
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
    
    with fish_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Upload Image Files**")
            uploaded_fish_image = st.file_uploader(
                "Choose FISH image file", 
                type=['png', 'jpg', 'jpeg'],
                key="fish_image"
            )
           
        with col2:
            st.markdown("**Upload Ground Truth Files**")
            uploaded_fish_gt = st.file_uploader(
                "Choose FISH Ground Truth file", 
                type=['png', 'jpg', 'jpeg'],
                key="fish_gt"
            )
        
        if uploaded_fish_image is not None:
            # Load image from uploaded file
            im = load_uploaded_image(uploaded_fish_image)
            
            if im is not None:
                st.image(im, caption=f"Citra FISH RGB Asli - {uploaded_fish_image.name}", width=500)
             
                st.subheader("1. Histogram Analysis: 3 channel RGB")
                r, g, b = plot_rgb_histogram(im, key_prefix="fish")  
                st.subheader("2. Preprocessing: CLAHE")
                b_clahe = apply_clahe(b, label="CLAHE on Blue Channel (FISH)", key_prefix="fish_b")
                
                st.subheader("3. Segmentasi: Otsu Thresholding")
                binary_mask = apply_otsu_threshold(b_clahe, title="Threshold FISH", is_dish=False)
                
                st.subheader("4. Filtering dan Pelabelan")
                image_segmented, labels, nlabels = filter_and_label_cells(binary_mask, key_prefix="fish")
                
                st.subheader("5. Watershed Segmentation")
                labels_ws, boundaries_ws, overlay_img = watershed_segmentation(image_segmented, im, key_prefix="fish")
                
                st.subheader("6. Evaluasi Citra terhadap Ground Truth")
                if uploaded_fish_gt is not None:
                   
                    gt_image = load_uploaded_image(uploaded_fish_gt)
                    if gt_image is not None:
                        gt_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)
                        evaluate_segmentation_result(gt_gray, labels_ws > 0)
                    else:
                        st.error("Error loading ground truth image")
                else:
                    st.warning("Please upload a ground truth image for evaluation")
                
                st.header("**ANALISIS KLASIFIKASI SINYAL HER2 DAN CEN17**")
                st.subheader("1. Plot channel tiap sinyal")
                her2_channel, cen17_channel = plot_chan_signal(im)
                
                st.subheader("2. Preprocessing : CLAHE")
                her2_clahe = apply_clahe_sig(her2_channel, label="HER2 (Red)", key_prefix="her2")
                cen17_clahe = apply_clahe_sig(cen17_channel, label="CEN17 (Green)", key_prefix="cen17")
                
                st.subheader("3. Stretching aim to get the signal")
                her2_stretched = stretch_channel(her2_channel, label="HER2", cmap="Reds", key_prefix="her2")
                cen17_stretched = stretch_channel(cen17_channel, label="CEN17", cmap="Greens", key_prefix="cen17")
                
                st.subheader("4. Deteksi Titik Sinyal dari Stretching")
                her2_coordsF, cen17_coordsF = detect_and_plot_signal_coords(im, her2_stretched, cen17_stretched, key_prefix="fish")
                
                st.subheader("5. Analisis Rasio HER2/CEN17 per Sel")
                df_ratio = analyze_her2_cen17_ratio(labels_ws, her2_coordsF, cen17_coordsF)
                
                st.subheader("6. Visualisasi Hasil Klasifikasi HER2/CEN17")
                visualize_her2_cen17_classification(im, labels_ws, her2_coordsF, cen17_coordsF, df_ratio)
            else:
                st.error("Error loading FISH image")
        else:
            st.warning("Please upload a FISH image file to start analysis")

    with dish_tab:
        st.subheader("Dual In Situ Hybridization (DISH)")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_dish_image = st.file_uploader(
                "Choose DISH image file", 
                type=['png', 'jpg', 'jpeg'],
                key="dish_image"
            )
        
        with col2:
            uploaded_dish_gt = st.file_uploader(
                "Choose DISH Ground Truth file", 
                type=['png', 'jpg', 'jpeg'],
                key="dish_gt"
            )
            
        if uploaded_dish_image is not None:
            im2 = load_uploaded_image(uploaded_dish_image)
            
            if im2 is not None:
                st.image(im2, caption=f"Citra DISH RGB Asli - {uploaded_dish_image.name}", width=500)
                
                st.subheader("1. Histogram Analysis: 3 channel RGB")
                rD, gD, bD = plot_rgb_histogram(im2, key_prefix="dish") 
                
                st.subheader("2. Preprocessing: CLAHE")
                r_clahe = apply_clahe(rD, label="CLAHE on Red Channel (DISH)", key_prefix="dish_r")
                
                st.subheader("3. Segmentasi: Otsu Thresholding")
                binary_mask_D = apply_otsu_threshold(r_clahe, title="Threshold DISH", is_dish=True)
                
                st.subheader("4. Filtering dan Pelabelan")
                image_segmented_D, labelsD, nlabelsD = filter_and_label_cells(binary_mask_D, key_prefix="dish")
                
                st.subheader("5. Watershed Segmentation")
                labels_ws_D, boundaries_ws_D, overlay_img_D = watershed_segmentation(image_segmented_D, im2, key_prefix="dish")
                
                st.subheader("6. Evaluasi Citra terhadap Ground Truth")
                if uploaded_dish_gt is not None:
                    # Load ground truth image
                    gt_image_D = load_uploaded_image(uploaded_dish_gt)
                    if gt_image_D is not None:
     
                        gt_gray_D = cv2.cvtColor(gt_image_D, cv2.COLOR_RGB2GRAY)
               
                        evaluate_segmentation_result(gt_gray_D, labels_ws_D > 0)
                    else:
                        st.error("Error loading DISH ground truth image")
                else:
                    st.warning("Please upload a ground truth image for evaluation")
                
                st.header("**ANALISIS KLASIFIKASI SINYAL HER2 DAN CEN17 PADA CITRA DISH**")
                st.subheader("1. Plot channel tiap sinyal")
                her2_channelD, cen17_channelD = extract_and_plot_dish_signal_channels(im2, cen17_mask_thresh=0.1)
                
                st.subheader("2. Stretching aim to get the signal")
                cen17_stretchedD = stretch_channel(her2_channelD, label="CEN17 DISH", cmap="Reds", key_prefix="her2D")
                her2_stretchedD = stretch_channel(cen17_channelD, label="HER2 DISH", cmap="Greens", key_prefix="cen17D")
                
                st.subheader("3. Deteksi Titik Sinyal dari Stretching DISH")
                her2_coordsD, cen17_coordsD = detect_and_plot_signal_coords(im2, her2_stretchedD, cen17_stretchedD, key_prefix="dish")
                
                st.subheader("4. Analisis Rasio HER2/CEN17 per Sel")
                df_ratioD = analyze_her2_cen17_ratio(labels_ws_D, her2_coordsD, cen17_coordsD)
                
                st.subheader("5. Visualisasi Hasil Klasifikasi HER2/CEN17")
                visualize_her2_cen17_classification(im2, labels_ws_D, her2_coordsD, cen17_coordsD, df_ratioD)
            else:
                st.error("Error loading DISH image")
        else:
            st.warning("Please upload a DISH image file to start analysis")
    
    st.markdown("---")
    st.subheader("🔬 FISH vs DISH Comparison")
    
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        st.markdown("### FISH (Fluorescence In Situ Hybridization)")
        st.markdown("""
        **Advantages:**
        - Higher sensitivity and specificity
        - Better signal-to-noise ratio
        - More precise quantification
        - Standard method for HER2 testing
        
        **Characteristics:**
        - Fluorescent probes (Red: HER2, Green: CEN17)
        - Requires fluorescence microscopy
        - DAPI nuclear counterstain (Blue)
        - Automated analysis available
        """)
    
    with comparison_col2:
        st.markdown("### DISH (Dual In Situ Hybridization)")
        st.markdown("""
        **Advantages:**
        - Uses standard brightfield microscopy
        - Permanent slides (no fading)
        - Cost-effective
        - Can be reviewed by pathologists
        
        **Characteristics:**
        - Chromogenic detection (Brown: HER2, Blue: CEN17)
        - Standard H&E-like appearance
        - Easier integration with routine workflow
        - Manual counting often required
        """)
    
    # Clinical guidelines
    st.markdown("---")
    st.subheader("📋 Clinical Guidelines")
    
    guidelines_col1, guidelines_col2 = st.columns(2)
    
    with guidelines_col1:
        st.markdown("### HER2 Testing Guidelines")
        st.markdown("""
        **ASCO/CAP Guidelines:**
        - HER2-Positive: Ratio ≥ 2.0
        - Equivocal: Ratio 1.8-2.0
        - HER2-Negative: Ratio < 1.8
        
        **Quality Requirements:**
        - Count minimum 20 cells
        - Ensure adequate signal intensity
        - Proper controls required
        """)
    
    with guidelines_col2:
        st.markdown("### Clinical Actions")
        st.markdown("""
        **HER2-Positive:**
        - Trastuzumab (Herceptin) therapy
        - Pertuzumab combination
        - T-DM1 for resistance
        
        **Equivocal Results:**
        - Repeat testing recommended
        - Consider alternative methods
        - Clinical correlation required
        """)

    st.markdown("---")
    st.markdown("*FISH and DISH Analyzer - Advanced Medical Image Analysis Tool*")

st.markdown("---")
st.caption("Medical Image Analyzer - Created by Yohanes")
