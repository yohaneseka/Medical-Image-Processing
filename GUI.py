import streamlit as st #library utama untuk membangun GUI
import numpy as np # untuk perhitungan numerik
import matplotlib.pyplot as plt #untuk membuat grafik
import cv2 #pemrosesan citra
import time #untuk mengukur waktu eksekusi
from skimage import io, color #untuk konversi gambar RGB ke grayscale
from io import BytesIO #untuk membaca gambar yang diunggah
from PIL import Image #untuk membaca gambar yang diunggah

st.set_page_config(
    page_title="Edge Detection Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Edge Detection Method Analyzer")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    st.subheader("Select Edge Detection Methods")
    prewitt_enabled = st.checkbox("Prewitt", value=True)
    sobel_enabled = st.checkbox("Sobel", value=True)
    roberts_enabled = st.checkbox("Roberts", value=True)
    extended_sobel_enabled = st.checkbox("Extended Sobel", value=True)
    kirsch_enabled = st.checkbox("Kirsch", value=True)
    
    st.subheader("Analysis Options")
    show_histograms = st.checkbox("Show Histograms", value=True)
    show_timing = st.checkbox("Show Timing Comparison", value=True)
    show_metrics = st.checkbox("Show Statistical Metrics", value=True)

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

def generate_figure_data(figure):
    buf = BytesIO()
    figure.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    return buf

# Main application logic
if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
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
        st.warning("Please select at least one edge detection method from the sidebar.")
else:
    # Display sample message when no image is uploaded
    st.info("Please upload an image using the sidebar to start the analysis.")

# Footer
st.markdown("---")
st.caption("Edge Detection Analyzer - Created by Yohanes")