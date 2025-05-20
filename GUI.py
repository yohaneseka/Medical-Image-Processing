import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import time
from skimage import color, filters
from io import BytesIO
from PIL import Image

st.set_page_config(
    page_title="Edge Detection Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Edge Detection Method Analyzer")

# Create sidebar tabs for Task 1 and Task 2
task_choice = st.sidebar.radio("Select Task", ["Task 1: Original Methods", "Task 2: Gaussian and Sharpening", "Task 3: Corner, Line, and Circle Detection"])

# ===================== TASK 1: ORIGINAL CODE =====================
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
    
    # File uploader outside tabs to be used by all detection methods
    uploaded_file_task3 = st.file_uploader("Upload an image for advanced analysis", type=["jpg", "jpeg", "png"], key="task3_uploader")
    
    # Create tabs for different detection methods
    tab1, tab2, tab3 = st.tabs(["Corner Detection", "Line Detection", "Circle Detection"])   
    
    # Process uploaded image if available
    if uploaded_file_task3 is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file_task3.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        with tab1:
            st.subheader("Corner Detection")
            st.write("Adjust parameters for Corner Detection")
            
            # Parameters for corner detection
            a = st.slider("a Value", min_value=0.01, max_value=0.5, value=0.04, step=0.01)
            threshold = st.slider("Threshold Value", min_value=100, max_value=10000000, value=101000, step=1000)
            
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
            
            # Process and display results
            start_manual = time.time()
            result_manual = harris_manual(gray, a=a, threshold=threshold)
            end_manual = time.time()
            
            start_lib = time.time()
            result_lib = harris_library(gray, a=a, threshold=0.01)  # Using relative threshold
            end_lib = time.time()
            
            time_manual = end_manual - start_manual
            time_lib = end_lib - start_lib
            
            # Display original image
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
            st.write("Line detection functionality will be implemented here")
            # Add your line detection code here
            
        with tab3:
            st.subheader("Circle Detection")
            st.write("Circle detection functionality will be implemented here")
            # Add your circle detection code here
    else:
        # Display message when no image is uploaded
        for tab in [tab1, tab2, tab3]:
            with tab:
                st.write("Please upload an image to perform detection.")
                
# Footer (shown on all tabs)
st.markdown("---")
st.caption("Edge Detection Analyzer - Created by Yohanes")
