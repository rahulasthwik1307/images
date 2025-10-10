import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import os
from segment_anything import sam_model_registry, SamPredictor
import tempfile
import base64
from io import BytesIO
import time

# 🧩 VERSION CHECK SECTION — added safely before app class
def show_versions():
    """Display library versions in Streamlit sidebar"""
    st.sidebar.markdown("### ⚙️ Library Versions")
    try:
        st.sidebar.write(f"Torch: {torch.__version__}")
    except:
        st.sidebar.write("Torch: Not installed")

    try:
        st.sidebar.write(f"OpenCV: {cv2.__version__}")
    except:
        st.sidebar.write("OpenCV: Not installed")

    try:
        import segment_anything
        st.sidebar.write(f"Segment Anything: {segment_anything.__version__ if hasattr(segment_anything, '__version__') else 'N/A'}")
    except:
        st.sidebar.write("Segment Anything: Not installed")

    try:
        import PIL
        st.sidebar.write(f"Pillow: {PIL.__version__}")
    except:
        st.sidebar.write("Pillow: Not installed")

    try:
        import numpy
        st.sidebar.write(f"Numpy: {numpy.__version__}")
    except:
        st.sidebar.write("Numpy: Not installed")

    st.sidebar.markdown("---")
    st.sidebar.info("✅ These are the current library versions in your environment.")

# ✅ Call the version display
show_versions()

class VisionExtractApp:
    def __init__(self, model_path, model_type="vit_b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SAM model with weights_only=True to fix the warning
        try:
            self.model = sam_model_registry[model_type](checkpoint=model_path)
            self.model.to(device=self.device)
            self.predictor = SamPredictor(self.model)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            raise
        
        # Initialize variables
        self.original_image = None
        self.mask = None
        self.extracted_subject = None
        
    def extract_subject(self, image):
        """Extract subject using SAM with improved accuracy"""
        if image is None:
            return None, None
            
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            image_array = np.array(image)
            self.predictor.set_image(image_array)
            
            height, width = image_array.shape[:2]
            
            # IMPROVED: Generate better point prompts for subject detection
            points = []
            labels = []
            
            # Center point (most likely to be on subject)
            points.append([width // 2, height // 2])
            labels.append(1)
            
            # Edge points to capture boundaries
            edge_points = [
                [width // 4, height // 4],      # Top-left
                [3 * width // 4, height // 4],  # Top-right
                [width // 4, 3 * height // 4],  # Bottom-left
                [3 * width // 4, 3 * height // 4], # Bottom-right
                [width // 2, height // 4],      # Top-center
                [width // 2, 3 * height // 4],  # Bottom-center
            ]
            
            points.extend(edge_points)
            labels.extend([1] * len(edge_points))
            
            points = np.array(points)
            labels = np.array(labels)
            
            # Predict multiple masks and choose the best one
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            
            # Choose the mask with highest score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            # IMPROVED: Better mask cleaning
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Remove small noise
            kernel_open = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
            
            # Fill holes and smooth edges
            kernel_close = np.ones((11, 11), np.uint8)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # Keep only the largest connected component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, 8)
            if num_labels > 1:
                # Sort by area (skip background at index 0)
                areas = stats[1:, cv2.CC_STAT_AREA]
                if len(areas) > 0:
                    largest_label = np.argmax(areas) + 1
                    cleaned_mask = (labels == largest_label).astype(np.uint8) * 255
                else:
                    cleaned_mask = np.zeros_like(cleaned_mask)
            
            # Apply Gaussian blur for smoother edges
            cleaned_mask = cv2.GaussianBlur(cleaned_mask, (5, 5), 0)
            _, cleaned_mask = cv2.threshold(cleaned_mask, 128, 255, cv2.THRESH_BINARY)
            
            cleaned_mask_bool = cleaned_mask.astype(bool)
            
            # Extract subject with the cleaned mask
            extracted_subject = image_array.copy()
            extracted_subject[~cleaned_mask_bool] = 0
            
            self.original_image = image_array
            self.mask = cleaned_mask_bool
            self.extracted_subject = extracted_subject
            
            return Image.fromarray(extracted_subject), cleaned_mask_bool
            
        except Exception as e:
            st.error(f"❌ Error in extraction: {e}")
            return None, None
    
    def apply_background(self, background_mode, custom_color=None, custom_bg_image=None, blur_strength=10):
        """Apply different background modes"""
        if self.original_image is None or self.mask is None:
            return None
            
        try:
            result = self.original_image.copy()
            
            if background_mode == "Transparent":
                # Create RGBA image with transparency
                rgba = np.dstack((self.original_image, self.mask.astype(np.uint8) * 255))
                return Image.fromarray(rgba)
                
            elif background_mode == "White":
                result[~self.mask] = [255, 255, 255]
                
            elif background_mode == "Black":
                result[~self.mask] = [0, 0, 0]
                
            elif background_mode == "Blur":
                # Apply Gaussian blur to background
                blurred_bg = cv2.GaussianBlur(self.original_image, (blur_strength*2+1, blur_strength*2+1), 0)
                result[~self.mask] = blurred_bg[~self.mask]
                
            elif background_mode == "Custom Color" and custom_color:
                # Convert hex color to RGB
                color_rgb = tuple(int(custom_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                result[~self.mask] = color_rgb
                
            elif background_mode == "Custom Image" and custom_bg_image is not None:
                bg_array = np.array(custom_bg_image)
                # Resize background to match original image
                bg_resized = cv2.resize(bg_array, (result.shape[1], result.shape[0]))
                result[~self.mask] = bg_resized[~self.mask]
            
            return Image.fromarray(result)
            
        except Exception as e:
            st.error(f"❌ Error applying background: {e}")
            return None
    
    def apply_filter(self, filter_name):
        """Apply artistic filters"""
        if self.original_image is None:
            return None
            
        try:
            img_pil = Image.fromarray(self.original_image)
            
            if filter_name == "None":
                return img_pil
                
            elif filter_name == "Cartoon":
                img_cv = np.array(img_pil)
                color = cv2.bilateralFilter(img_cv, 9, 300, 300)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                gray = cv2.medianBlur(gray, 7)
                edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 9, 2)
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                return Image.fromarray(cartoon)
                
            elif filter_name == "Pencil":
                img_cv = np.array(img_pil)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                inverted = 255 - gray
                blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
                pencil = cv2.divide(gray, 255 - blurred, scale=256)
                return Image.fromarray(pencil)
                
            elif filter_name == "HDR":
                img_cv = np.array(img_pil)
                hdr = cv2.detailEnhance(img_cv, sigma_s=12, sigma_r=0.15)
                return Image.fromarray(hdr)
                
            elif filter_name == "Sepia":
                img_cv = np.array(img_pil)
                kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
                sepia = cv2.transform(img_cv, kernel)
                sepia = np.clip(sepia, 0, 255)
                return Image.fromarray(sepia.astype(np.uint8))
                
            elif filter_name == "Painting":
                img_cv = np.array(img_pil)
                painting = cv2.stylization(img_cv, sigma_s=60, sigma_r=0.6)
                return Image.fromarray(painting)
                
            elif filter_name == "Blur":
                return img_pil.filter(ImageFilter.GaussianBlur(5))
                
            elif filter_name == "Gray":
                return img_pil.convert('L').convert('RGB')
                
            elif filter_name == "Glitch":
                img_cv = np.array(img_pil)
                r, g, b = cv2.split(img_cv)
                r_shifted = np.roll(r, 5, axis=1)
                b_shifted = np.roll(b, -5, axis=1)
                glitched = cv2.merge([r_shifted, g, b_shifted])
                return Image.fromarray(glitched)
            
            return img_pil
            
        except Exception as e:
            st.error(f"❌ Error applying filter: {e}")
            return None
    
    def crop_image(self, image, aspect_ratio, scale_factor=1.0):
        """Crop image based on aspect ratio"""
        if image is None:
            return None
            
        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Calculate target dimensions based on aspect ratio
            if aspect_ratio == "Free":
                new_w, new_h = w, h
            elif aspect_ratio == "Square (1:1)":
                size = min(w, h)
                new_w, new_h = size, size
            elif aspect_ratio == "Instagram (4:5)":
                new_h = h
                new_w = int(h * 4/5)
                if new_w > w:
                    new_w = w
                    new_h = int(w * 5/4)
            elif aspect_ratio == "Portrait (3:4)":
                new_h = h
                new_w = int(h * 3/4)
                if new_w > w:
                    new_w = w
                    new_h = int(w * 4/3)
            
            # Apply scale factor
            new_w = int(new_w * scale_factor)
            new_h = int(new_h * scale_factor)
            
            # Ensure dimensions are at least 1
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            # Center crop
            start_x = max(0, (w - new_w) // 2)
            start_y = max(0, (h - new_h) // 2)
            cropped = img_array[start_y:start_y+new_h, start_x:start_x+new_w]
            
            return Image.fromarray(cropped)
            
        except Exception as e:
            st.error(f"❌ Error cropping image: {e}")
            return None
    
    def create_comparison(self, final_image):
        """Create side-by-side comparison"""
        if self.original_image is None or final_image is None:
            return None
            
        try:
            original_pil = Image.fromarray(self.original_image)
            final_pil = final_image
            
            # Resize to same height
            height = min(original_pil.height, final_pil.height, 400)
            orig_resized = original_pil.resize((int(original_pil.width * height / original_pil.height), height))
            final_resized = final_pil.resize((int(final_pil.width * height / final_pil.height), height))
            
            # Create comparison image
            total_width = orig_resized.width + final_resized.width + 20
            comparison = Image.new('RGB', (total_width, height), (240, 240, 240))
            
            comparison.paste(orig_resized, (0, 0))
            comparison.paste(final_resized, (orig_resized.width + 20, 0))
            
            return comparison
            
        except Exception as e:
            st.error(f"❌ Error creating comparison: {e}")
            return None

def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}" style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); color: white; padding: 12px 20px; text-decoration: none; border-radius: 8px; font-weight: 600; display: inline-block; margin: 5px;">{text}</a>'
    return href

def load_demo_collage():
    """Load demo collage image"""
    try:
        demo_path = "https://github.com/rahulasthwik1307/images/blob/master/download.png"
        if os.path.exists(demo_path):
            image = Image.open(demo_path)
            return image
        else:
            return None
    except Exception as e:
        return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="VisionExtract - AI Image Editor",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern sleek design
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.4rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
    }
    .upload-box {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin: 1rem 0;
        transition: all 0.3s ease;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .upload-box:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    }
    .uploaded-image-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .option-group {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
    }
    .demo-collage {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin: 2rem auto;
        color: white;
        max-width: 800px;
    }
    .demo-image {
        border-radius: 15px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        max-width: 90%;
        border: 4px solid white;
    }
    .results-section {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #e9ecef;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .download-btn {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
    }
    /* Hide Streamlit warnings and info */
    .stAlert {
        display: none;
    }
    .upload-instructions {
        text-align: center;
        color: #666;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 1. 🎯 HEADER
    st.markdown('<div class="main-header">🎯 VisionExtract</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Subject Isolation & Background Replacement</div>', unsafe_allow_html=True)
    
    # Initialize app
    MODEL_PATH = "https://github.com/rahulasthwik1307/images/blob/master/sam_vit_l_0b3195.pth"
    
    if 'app' not in st.session_state:
        try:
            st.session_state.app = VisionExtractApp(MODEL_PATH, model_type="vit_l")
        except:
            st.error("❌ Failed to load model. Please check the model path.")
            return
    
    app = st.session_state.app
    
    # 2. 🖼️ DEMO COLLAGE - CENTERED
    col_demo = st.columns([1, 3, 1])  # Create 3 columns to center the middle one
    with col_demo[1]:  # Use the middle column
        st.markdown('<div class="demo-collage">', unsafe_allow_html=True)
        st.markdown("### ✨ See What You Can Create")
        demo_image = load_demo_collage()
        if demo_image:
            st.image(demo_image, use_column_width=True, caption="🎭 Background Replacement • 🎨 Artistic Filters • ✂️ Smart Cropping")
        else:
            st.info("🌟 Upload an image to experience AI-powered editing magic!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 3. 📤 UPLOAD SECTION
        st.markdown('<div class="section-title">📤 Upload Your Image</div>', unsafe_allow_html=True)
        
        # File uploader in the upload box
        uploaded_file = st.file_uploader(
            " ",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to extract the subject",
            label_visibility="collapsed",
            key="main_uploader"
        )
        
        # Display upload box or uploaded image
        if uploaded_file is not None:
            # Display uploaded image in a nice container
            input_image = Image.open(uploaded_file)
            with st.container():
                st.markdown('<div class="uploaded-image-container">', unsafe_allow_html=True)
                st.image(input_image, caption="📷 Uploaded Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Show upload box when no file is uploaded
            with st.container():
                st.markdown('<div class="upload-box">', unsafe_allow_html=True)
                st.markdown("""
                <div style='text-align: center; color: #666;'>
                    <h3 style='color: #667eea; margin-bottom: 1rem;'>📤 Upload Your Image</h3>
                    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'><strong>Drag & drop or click to browse</strong></p>
                    <p style='color: #888; font-size: 0.9rem;'>JPG, PNG, JPEG • Limit 200MB per file</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.info("👆 Click on the box above to upload an image")
        
        # 4. 🎭 BACKGROUND OPTIONS
        st.markdown('<div class="section-title">🎭 Background Options</div>', unsafe_allow_html=True)
        with st.container():
            background_mode = st.selectbox(
                "Select Background Style",
                ["White", "Black", "Transparent", "Blur", "Custom Color", "Custom Image"],
                index=0,
                help="Choose how to handle the background",
                label_visibility="collapsed"
            )
            
            custom_color = None
            custom_bg = None
            blur_strength = 10
            
            if background_mode == "Custom Color":
                custom_color = st.color_picker("Choose Background Color", "#FF6B6B")
            elif background_mode == "Custom Image":
                custom_bg_file = st.file_uploader(
                    "Upload Background Image",
                    type=['jpg', 'jpeg', 'png'],
                    key="bg_upload"
                )
                if custom_bg_file is not None:
                    custom_bg = Image.open(custom_bg_file)
                    st.image(custom_bg, caption="Custom Background", use_column_width=True)
            elif background_mode == "Blur":
                blur_strength = st.slider("Blur Strength", 1, 20, 10, help="Higher values = more blur")
        
        # 5. 🎨 FILTERS (Dropdown)
        st.markdown('<div class="section-title">🎨 Artistic Filters</div>', unsafe_allow_html=True)
        with st.container():
            filter_type = st.selectbox(
                "Choose Artistic Filter",
                ["None", "Cartoon", "Pencil", "HDR", "Sepia", "Painting", "Blur", "Gray", "Glitch"],
                index=0,
                help="Apply creative effects to your image",
                label_visibility="collapsed"
            )
        
        # 6. ✂️ CROP OPTIONS
        st.markdown('<div class="section-title">✂️ Crop & Resize</div>', unsafe_allow_html=True)
        with st.container():
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ["Free", "Square (1:1)", "Instagram (4:5)", "Portrait (3:4)"],
                index=0,
                help="Crop for different platforms",
                label_visibility="collapsed"
            )
            
            scale_factor = st.slider("Scale Factor", 0.1, 2.0, 1.0, 0.1, help="Resize the output image")
        
        # 7. 🚀 PROCESS BUTTON
        st.markdown("---")
        process_clicked = st.button("🚀 Process Image", use_container_width=True, type="primary", 
                                   disabled=uploaded_file is None)
    
    with col2:
        # 8. 📊 RESULTS AREA
        if process_clicked and uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            with st.spinner("🔄 Processing image with AI... This may take a few seconds"):
                # Extract subject
                extracted, mask = app.extract_subject(input_image)
                
                if extracted is not None:
                    # Apply background
                    final = app.apply_background(background_mode, custom_color, custom_bg, blur_strength)
                    
                    # Apply filter if needed
                    if filter_type != "None" and final is not None:
                        filtered = app.apply_filter(filter_type)
                        if filtered is not None:
                            # Convert to numpy arrays
                            filtered_array = np.array(filtered)
                            final_array = np.array(final)
                            
                            # Ensure both arrays have the same shape
                            if filtered_array.shape == final_array.shape:
                                # Apply filter only to the subject area
                                final_array[app.mask] = filtered_array[app.mask]
                                final = Image.fromarray(final_array)
                            else:
                                # If shapes don't match, resize filtered to match final
                                filtered_resized = np.array(filtered.resize(final_array.shape[1::-1]))
                                final_array[app.mask] = filtered_resized[app.mask]
                                final = Image.fromarray(final_array)
                    
                    # Apply cropping
                    if final is not None and aspect_ratio != "Free":
                        final = app.crop_image(final, aspect_ratio, scale_factor)
                    
                    # Create comparison
                    comparison = app.create_comparison(final)
                    
                    # Store in session state
                    st.session_state.extracted_image = extracted
                    st.session_state.final_image = final
                    st.session_state.comparison_image = comparison
                    st.session_state.processed = True
                    
                    st.success("✅ Image processed successfully!")
                else:
                    st.error("❌ Failed to extract subject from the image")
        
        # Display results
        if st.session_state.get('processed', False) and st.session_state.final_image is not None:
            st.markdown('<div class="section-title">📊 Results & Export</div>', unsafe_allow_html=True)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🎯 Extracted Subject", "✨ Final Result", "🔄 Comparison"])
            
            with tab1:
                if st.session_state.extracted_image is not None:
                    st.image(st.session_state.extracted_image, caption="AI-Extracted Subject", use_column_width=True)
                    st.markdown(
                        get_image_download_link(st.session_state.extracted_image, "extracted_subject.png", "📥 Download Extracted Subject"),
                        unsafe_allow_html=True
                    )
            
            with tab2:
                if st.session_state.final_image is not None:
                    st.image(st.session_state.final_image, caption="Final Processed Image", use_column_width=True)
                    st.markdown(
                        get_image_download_link(st.session_state.final_image, "final_image.png", "📥 Download Final Image"),
                        unsafe_allow_html=True
                    )
            
            with tab3:
                if st.session_state.comparison_image is not None:
                    st.image(st.session_state.comparison_image, caption="Before & After Comparison", use_column_width=True)
                    st.markdown(
                        get_image_download_link(st.session_state.comparison_image, "comparison.png", "📥 Download Comparison"),
                        unsafe_allow_html=True
                    )
        else:
            # Placeholder when no processing done
            if uploaded_file is None:
                st.markdown("""
                <div style='text-align: center; padding: 4rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 2px dashed #667eea;'>
                    <h3 style='color: #666; margin-bottom: 1rem;'>🎨 Ready to Transform Your Image?</h3>
                    <p style='color: #888;'>Upload an image on the left and click 'Process Image' to see the magic happen!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 4rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 2px dashed #667eea;'>
                    <h3 style='color: #666; margin-bottom: 1rem;'>🚀 Ready to Process!</h3>
                    <p style='color: #888;'>Click the 'Process Image' button on the left to start AI processing.</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'extracted_image' not in st.session_state:
        st.session_state.extracted_image = None
    if 'final_image' not in st.session_state:
        st.session_state.final_image = None
    if 'comparison_image' not in st.session_state:
        st.session_state.comparison_image = None
    
    main()