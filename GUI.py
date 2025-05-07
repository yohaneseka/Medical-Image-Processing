import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
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
task_choice = st.sidebar.radio("Select Task", ["Task 1", "Task 2"])

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
elif task_choice == "Task 2: Advanced Analysis":
    st.header("Task 2: Advanced Edge Detection Analysis")
    
    # Task 2 settings
    st.subheader("Canny Edge Detection Pipeline Settings")
    uploaded_file_task2 = st.file_uploader("Upload an image for advanced analysis", type=["jpg", "jpeg", "png"], key="task2_uploader")
    
    # Advanced parameters
    st.write("Adjust parameters for the Canny edge detection pipeline:")
    col1, col2 = st.columns(2)
    
    with col1:
        sigma = st.slider("Gaussian Blur Sigma", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                         help="Controls the amount of blur. Higher values create more blur.")
        high_threshold_factor = st.slider("High Threshold Factor", min_value=0.05, max_value=0.3, value=0.15, step=0.01,
                                        help="Fraction of maximum magnitude to use as high threshold. Higher values detect fewer edges.")
    
    with col2:
        low_threshold_factor = st.slider("Low Threshold Factor", min_value=0.01, max_value=0.99, value=0.05, step=0.01,
                                       help="Fraction of high threshold to use as low threshold. Controls edge connectivity.")
        show_all_steps = st.checkbox("Show All Processing Steps", value=True,
                                  help="Display every step of the Canny edge detection pipeline")
    
    # Separator between settings and results
    st.markdown("---")
    
    def apply_canny_pipeline(image):
        """
        Apply full Canny edge detection pipeline with visualizations of each step
        """
        start_time = time.time()
        
        # Convert image to grayscale if it's color
        if len(image.shape) > 2:
            gray_image = color.rgb2gray(image)
            gray_image = (gray_image * 255).astype(np.uint8)
        else:
            gray_image = image
        
        rows, columns = gray_image.shape
        
        # Step 2: Apply Gaussian blur
        gaussian_image = filters.gaussian(gray_image, sigma=sigma, preserve_range=True).astype(np.uint8)
        
        # Step 3: Apply Sobel for edge detection (fx and fy)
        fx = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0, ksize=3)
        fy = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Step 4: Calculate Magnitude and Angle
        magnitude = np.sqrt(fx**2 + fy**2)
        magnitude_normalized = (magnitude / magnitude.max() * 255).astype(np.uint8)
        angle = np.arctan2(fy, fx)
        
        # Step 5: Create color visualization based on edge direction
        hsv = np.zeros((rows, columns, 3), dtype=np.uint8)
        hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue from angle
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = np.minimum(magnitude_normalized, 255)  # Value from magnitude
        color_edges = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Step 6: Non-maximum Suppression
        def non_max_suppression(magnitude, angle):
            result = np.zeros_like(magnitude)
            angle = angle * 180. / np.pi
            angle[angle < 0] += 180
            
            for i in range(1, rows - 1):
                for j in range(1, columns - 1):
                    # 0 degrees
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                    # 45 degrees
                    elif 22.5 <= angle[i, j] < 67.5:
                        neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                    # 90 degrees
                    elif 67.5 <= angle[i, j] < 112.5:
                        neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                    # 135 degrees
                    else:
                        neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                    
                    if magnitude[i, j] >= max(neighbors):
                        result[i, j] = magnitude[i, j]
            
            return result
        
        non_max_img = non_max_suppression(magnitude, angle)
        non_max_normalized = (non_max_img / non_max_img.max() * 255).astype(np.uint8) if non_max_img.max() > 0 else np.zeros_like(non_max_img, dtype=np.uint8)
        
        # Step 7: Double Thresholding
        high_threshold = magnitude.max() * high_threshold_factor
        low_threshold = high_threshold * low_threshold_factor
        
        strong_edges = (non_max_img > high_threshold)
        weak_edges = (non_max_img >= low_threshold) & (non_max_img <= high_threshold)
        threshold_img = np.zeros_like(non_max_img, dtype=np.uint8)
        threshold_img[strong_edges] = 255
        threshold_img[weak_edges] = 75
        
        # Step 8: Hysteresis - connecting weak edges to strong edges
        def hysteresis(img, weak=75, strong=255):
            output = np.zeros_like(img)
            output[img == strong] = strong
            
            # Identify weak pixels adjacent to strong pixels
            indices = np.transpose(np.nonzero(img == weak))
            for i, j in indices:
                # Check 8 neighbors
                if i > 0 and i < img.shape[0]-1 and j > 0 and j < img.shape[1]-1:
                    neighbors = img[i-1:i+2, j-1:j+2]
                    if strong in neighbors:
                        output[i, j] = strong
            
            return output
        
        hysteresis_img = hysteresis(threshold_img)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'gray_image': gray_image,
            'gaussian_image': gaussian_image,
            'fx': fx,
            'fy': fy,
            'magnitude': magnitude_normalized,
            'angle': angle,
            'color_edges': color_edges,
            'non_max_img': non_max_normalized,
            'threshold_img': threshold_img,
            'hysteresis_img': hysteresis_img,
            'execution_time': execution_time
        }
    
    # Advanced Results area
    if uploaded_file_task2 is not None:
        try:
            # Read the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file_task2.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
            # Process with Canny Pipeline
            with st.spinner("Processing image with Canny edge detection pipeline..."):
                results = apply_canny_pipeline(image)
            
            # Display execution time
            st.success(f"Processing completed in {results['execution_time']*1000:.2f} ms")
            
            # Display final result first
            st.subheader("Final Canny Edge Detection Result")
            col1, col2 = st.columns(2)
            with col1:
                st.image(results['hysteresis_img'], caption="After Hysteresis (Final Result)", use_column_width=True)
            with col2:
                st.image(results['color_edges'], caption="Edge Direction Colormap", use_column_width=True)
            
            # Display all steps if selected
            if show_all_steps:
                st.subheader("Complete Canny Edge Detection Pipeline")
                
                # Create a figure with subplots for visualization
                fig, axes = plt.subplots(2, 6, figsize=(20, 10))
                fig.suptitle('Canny Edge Detection Pipeline Steps', fontsize=16)
                
                # First row
                axes[0, 0].imshow(image)
                axes[0, 0].set_title('Original')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(results['gray_image'], cmap='gray')
                axes[0, 1].set_title('Grayscale')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(results['gaussian_image'], cmap='gray')
                axes[0, 2].set_title(f'Gaussian (σ={sigma})')
                axes[0, 2].axis('off')
                
                axes[0, 3].imshow(results['fx'], cmap='gray')
                axes[0, 3].set_title('Horizontal Edges (fx)')
                axes[0, 3].axis('off')
                
                axes[0, 4].imshow(results['fy'], cmap='gray')
                axes[0, 4].set_title('Vertical Edges (fy)')
                axes[0, 4].axis('off')
                
                axes[0, 5].imshow(results['magnitude'], cmap='gray')
                axes[0, 5].set_title('Edge Magnitude')
                axes[0, 5].axis('off')
                
                # Second row
                normalized_angle = ((results['angle'] + np.pi) / (2 * np.pi))
                axes[1, 0].imshow(normalized_angle, cmap='hsv')
                axes[1, 0].set_title('Edge Direction')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(results['color_edges'])
                axes[1, 1].set_title('Colorized Edges')
                axes[1, 1].axis('off')
                
                axes[1, 2].imshow(results['non_max_img'], cmap='gray')
                axes[1, 2].set_title('Non-max Suppression')
                axes[1, 2].axis('off')
                
                axes[1, 3].imshow(results['threshold_img'], cmap='gray')
                axes[1, 3].set_title(f'Double Threshold\n(H:{high_threshold_factor:.2f}, L:{low_threshold_factor:.2f})')
                axes[1, 3].axis('off')
                
                axes[1, 4].imshow(results['hysteresis_img'], cmap='gray')
                axes[1, 4].set_title('Hysteresis')
                axes[1, 4].axis('off')
                
                # Make one plot empty
                axes[1, 5].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Additional step-by-step explanation
                st.subheader("Understanding the Canny Edge Detection Pipeline")
                st.write("""
                The Canny edge detection algorithm is considered the optimal edge detector and follows these steps:
                
                1. **Noise Reduction**: The image is smoothed with a Gaussian filter to reduce noise.
                
                2. **Gradient Calculation**: Sobel filters compute the horizontal (fx) and vertical (fy) edge gradients.
                
                3. **Magnitude & Direction**: Edge magnitude and direction are calculated from the gradients.
                
                4. **Non-Maximum Suppression**: Edges are thinned by keeping only local maxima.
                
                5. **Double Thresholding**: Edges are classified as strong, weak, or non-edges using two thresholds.
                
                6. **Edge Tracking by Hysteresis**: Weak edges connected to strong edges are kept, others discarded.
                """)
                
                # Add parameter explanation
                st.subheader("Parameter Effects")
                st.write(f"""
                - **Sigma ({sigma})**: Controls the amount of blur. Higher values reduce noise but may lose details.
                
                - **High Threshold ({high_threshold_factor})**: Determines what is considered a strong edge. Higher values detect fewer edges.
                
                - **Low Threshold ({low_threshold_factor} × High Threshold)**: Determines what might be an edge. Lower values include more potential edges.
                """)
                
                # Performance metrics
                st.subheader("Edge Detection Metrics")
                metrics_data = {
                    "Parameter": ["Execution Time", "Edge Pixels (%)", "Strong Edge Pixels (%)", "Weak Edge Pixels (%)"],
                    "Value": [
                        f"{results['execution_time']*1000:.2f} ms",
                        f"{np.count_nonzero(results['hysteresis_img']) / results['hysteresis_img'].size * 100:.2f}%",
                        f"{np.count_nonzero(results['threshold_img'] == 255) / results['threshold_img'].size * 100:.2f}%",
                        f"{np.count_nonzero(results['threshold_img'] == 75) / results['threshold_img'].size * 100:.2f}%"
                    ]
                }
                st.table(metrics_data)
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.info("Please try adjusting the parameters or upload a different image.")
    else:
        st.info("Please upload an image to start the advanced edge detection analysis.")

# Footer (shown on all tabs)
st.markdown("---")
st.caption("Edge Detection Analyzer - Created by Yohanes")
