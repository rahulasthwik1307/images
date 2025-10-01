# app.py
import io
import os
from typing import Tuple

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet101

import streamlit as st

# ---------------------- USER CONFIG ----------------------
MODEL_PATH = r"C:\Users\rahul\Downloads\Telegram Desktop\deeplabv3_resnet101_stage3_full.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Demo image path
DEMO_IMAGE_PATH = r"C:\Users\rahul\Downloads\infosys project\segmentation_result.png"

# ---------------------- HELPERS & MODEL BUILD ----------------------
def replace_bn_with_gn(module, num_groups=32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features))
        else:
            replace_bn_with_gn(child, num_groups)

def build_model(num_classes=1):
    model = deeplabv3_resnet101(weights=None, aux_loss=False)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    replace_bn_with_gn(model)
    return model

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    model = build_model(num_classes=1)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_sd = model.state_dict()
    filtered = {k.replace("module.",""): v for k,v in sd.items() if k.replace("module.","") in model_sd and model_sd[k.replace("module.","")].shape==v.shape}
    model.load_state_dict(filtered, strict=False)
    model.to(DEVICE).eval()
    return model

# ---------------------- IMAGE + INFERENCE UTIL ----------------------
def resize_keep_aspect_and_pad(pil_img, min_short, scale=1.0, pad_fill=(0,0,0), max_side=1280):
    w, h = pil_img.size
    base_scale = min_short / float(min(w, h))
    tw, th = int(w*base_scale*scale), int(h*base_scale*scale)
    if max(tw,th) > max_side:
        ratio = max_side / float(max(tw,th))
        tw, th = int(tw*ratio), int(th*ratio)
    img_resized = pil_img.resize((tw, th), Image.BILINEAR)
    S = max(tw, th)
    canvas = Image.new("RGB",(S,S), pad_fill)
    l,t = (S-tw)//2, (S-th)//2
    canvas.paste(img_resized,(l,t))
    return canvas,(l,t,tw,th,S)

def tensor_from_pil(img):
    t = TF.to_tensor(img)
    return TF.normalize(t, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).unsqueeze(0)

def upsample_to(x,h,w):
    return F.interpolate(x,size=(h,w),mode="bilinear",align_corners=False)

def improved_adaptive_threshold(prob_map):
    vals = prob_map.flatten()
    m = vals.mean()
    s = vals.std()
    stat_thresh = max(0.15, min(0.7, m + 0.3 * s))
    p90 = np.percentile(vals, 90)
    p10 = np.percentile(vals, 10)
    percentile_thresh = p10 + 0.4 * (p90 - p10)
    hist, bins = np.histogram(vals, bins=128, range=(0, 1))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    high_prob_indices = hist > np.percentile(hist, 70)
    if np.any(high_prob_indices):
        high_prob_centers = bin_centers[high_prob_indices]
        high_prob_weights = hist[high_prob_indices]
        weighted_thresh = np.average(high_prob_centers, weights=high_prob_weights)
    else:
        weighted_thresh = 0.5
    final_thresh = (0.4 * stat_thresh + 0.4 * percentile_thresh + 0.2 * weighted_thresh)
    final_thresh = max(0.1, min(0.8, final_thresh))
    return final_thresh, float(m), float(s)

def improved_morphology(mask):
    if np.sum(mask) == 0:
        return mask
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    k_small = max(3, min(7, int(min(h, w) / 150)))
    k_medium = max(5, min(11, int(min(h, w) / 100)))
    k_small = k_small + 1 if k_small % 2 == 0 else k_small
    k_medium = k_medium + 1 if k_medium % 2 == 0 else k_medium
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_medium, k_medium))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return m
    max_area = np.max(areas)
    result = np.zeros_like(m)
    for i in range(1, num_labels):
        if areas[i-1] >= max_area * 0.08:
            result[labels == i] = 1
    return result

def improved_grabcut_refine(orig, mask, prob_map, iters=8):
    if np.sum(mask) < 15:
        return mask
    img = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    h, w = mask.shape
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc_mask[prob_map > 0.85] = cv2.GC_FGD
    gc_mask[(prob_map > 0.45) & (prob_map <= 0.85)] = cv2.GC_PR_FGD
    gc_mask[(prob_map >= 0.15) & (prob_map <= 0.45)] = cv2.GC_PR_BGD
    gc_mask[prob_map < 0.15] = cv2.GC_BGD
    fg_count = np.sum((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
    bg_count = np.sum((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
    if fg_count < 30 or bg_count < 30:
        gc_mask[mask == 1] = cv2.GC_PR_FGD
        gc_mask[mask == 0] = cv2.GC_PR_BGD
    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)
        refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0)
        return refined_mask.astype(np.uint8)
    except Exception:
        return mask

def run_one_pass(model, orig: Image.Image, input_size, scale=1.0, flip=False):
    canvas, (l, t, tw, th, S) = resize_keep_aspect_and_pad(orig, input_size, scale)
    if flip:
        canvas = TF.hflip(canvas)
    inp = tensor_from_pil(canvas).to(DEVICE)
    with torch.no_grad():
        out = model(inp)["out"]
        out = upsample_to(out, S, S)
        p = torch.sigmoid(out)[0, 0].cpu().numpy()
    if flip:
        p = np.fliplr(p)
    pc = p[t:t+th, l:l+tw]
    return cv2.resize(pc, (orig.width, orig.height), cv2.INTER_LINEAR)

def enhanced_tta_predict(model, orig: Image.Image):
    H, W = orig.height, orig.width
    acc = np.zeros((H, W), np.float32)
    c = 0
    tta_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    for s in tta_scales:
        for flip in [False, True]:
            prob_map = run_one_pass(model, orig, 640, s, flip)
            acc += prob_map
            c += 1
    avg = acc / c
    thr, mc, sc = improved_adaptive_threshold(avg)
    mask = (avg > thr).astype(np.uint8)
    mask = improved_morphology(mask)
    mask = improved_grabcut_refine(orig, mask, avg, iters=8)
    mask = improved_morphology(mask)
    orig_arr = np.array(orig).astype(np.uint8)
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.dstack([orig_arr, alpha])
    masked_pil = Image.fromarray(rgba)
    return masked_pil, avg, float(mc), float(sc)

def apply_custom_background(transparent_image, bg_color, bg_transparency):
    """
    Apply custom background to transparent image with proper transparency handling
    """
    # Convert RGBA to numpy array
    rgba_array = np.array(transparent_image)
    
    # Extract RGB and Alpha channels
    rgb = rgba_array[:, :, :3]
    alpha = rgba_array[:, :, 3] if rgba_array.shape[2] == 4 else np.ones(rgba_array.shape[:2], dtype=np.uint8) * 255
    
    # Convert hex color to RGB
    bg_color_rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Create background image
    bg_image = np.zeros_like(rgb)
    bg_image[:, :, 0] = bg_color_rgb[0]  # R
    bg_image[:, :, 1] = bg_color_rgb[1]  # G
    bg_image[:, :, 2] = bg_color_rgb[2]  # B
    
    # Normalize alpha to 0-1 range
    alpha_normalized = alpha.astype(float) / 255.0
    
    # Calculate transparency factor (0 = solid background, 100 = fully transparent)
    transparency_factor = bg_transparency / 100.0
    
    # Blend images
    if bg_transparency == 100:
        # Fully transparent - return original transparent image
        return transparent_image
    elif bg_transparency == 0:
        # Solid background - composite with solid color
        result = np.zeros_like(rgb)
        for i in range(3):
            result[:, :, i] = rgb[:, :, i] * alpha_normalized + bg_image[:, :, i] * (1 - alpha_normalized)
    else:
        # Semi-transparent background
        result = np.zeros_like(rgb)
        for i in range(3):
            # Blend: subject + (background with transparency)
            result[:, :, i] = (rgb[:, :, i] * alpha_normalized + 
                              bg_image[:, :, i] * (1 - alpha_normalized) * (1 - transparency_factor))
    
    return Image.fromarray(result.astype(np.uint8))

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="VisionExtract", page_icon="🛰️", layout="wide")

# CSS
st.markdown(
    """
    <style>
    .title { 
        font-size: 32px; 
        font-weight: 600; 
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle { 
        color: #6c757d; 
        text-align: center;
        margin-bottom: 30px;
    }
    .demo-container {
        text-align: center;
        margin: 20px 0;
    }
    .demo-caption {
        font-style: italic;
        color: #666;
        margin-top: 10px;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with demo image
st.markdown('<div class="title">VisionExtract: AI-Powered Background Removal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any image → AI removes background → Download with transparent or custom background</div>', unsafe_allow_html=True)

# Demo image
try:
    if os.path.exists(DEMO_IMAGE_PATH):
        demo_image = Image.open(DEMO_IMAGE_PATH)
        st.markdown('<div class="demo-container">', unsafe_allow_html=True)
        st.image(demo_image, use_column_width=True, caption="See how it works: Upload an image and get instant background removal!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Demo image not found. Please check the file path.")
except Exception as e:
    st.warning(f"Could not load demo image: {e}")

# Sidebar with simplified settings
with st.sidebar:
    st.header("🎨 Background Settings")
    
    # Background color picker
    bg_color = st.color_picker("Choose Background Color", "#FFFFFF")
    
    # Background transparency with better explanation
    bg_transparency = st.slider(
        "Background Visibility", 
        min_value=0, 
        max_value=100, 
        value=100,
        help="0% = Solid color background, 100% = Fully transparent background"
    )
    
    # Show current setting
    if bg_transparency == 100:
        st.success("✅ Fully transparent background")
    elif bg_transparency == 0:
        st.info("🎨 Solid color background")
    else:
        st.info(f"🌫️ Semi-transparent background ({bg_transparency}% visible)")
    
    st.markdown("---")
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Upload** an image with a clear subject
    2. **Click** 'Generate Mask' 
    3. **Adjust** background settings
    4. **Download** your result!
    
    💡 **Transparency Tips:**
    - 100% = No background (transparent PNG)
    - 0% = Solid color background  
    - 50% = See-through background
    """)

# Load model
with st.spinner("Loading AI model..."):
    model = load_model(MODEL_PATH)

# Main columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"], accept_multiple_files=False, label_visibility="collapsed")
    
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True, caption="Original Image")
        
        if st.button("🎭 Generate Mask", type="primary", use_container_width=True):
            try:
                with st.spinner("AI is removing background..."):
                    masked_pil, avg_prob_map, mc, sc = enhanced_tta_predict(model, image)
                    st.session_state["result_masked"] = masked_pil
                    st.session_state["original_image"] = image
                    st.session_state["conf_mean"] = mc
                    st.session_state["conf_std"] = sc
                    st.session_state["avg_prob_map"] = avg_prob_map
                    st.success("Background removed successfully!")
            except Exception as e:
                st.error(f"Error processing image: {e}")

with col2:
    st.markdown("### 🎯 Results")
    
    if "result_masked" in st.session_state:
        # Display confidence metrics
        mc = st.session_state.get("conf_mean", 0.0)
        sc = st.session_state.get("conf_std", 0.0)
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Confidence Score", f"{mc:.3f}")
        with col_metric2:
            st.metric("Quality", "High" if mc > 0.7 else "Medium" if mc > 0.5 else "Low")
        
        # Apply custom background
        final_image = apply_custom_background(
            st.session_state["result_masked"],
            bg_color,
            bg_transparency
        )
        
        # Create appropriate caption
        if bg_transparency == 100:
            caption = "Transparent Background (PNG)"
            bg_info = "No background - transparent"
        elif bg_transparency == 0:
            caption = f"Solid {bg_color} Background"
            bg_info = f"Solid {bg_color} background"
        else:
            caption = f"Semi-transparent {bg_color} Background ({bg_transparency}% visible)"
            bg_info = f"{bg_color} background with {bg_transparency}% visibility"
        
        # Display result with background info
        st.image(final_image, use_column_width=True, caption=caption)
        st.info(f"Current background: {bg_info}")
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Download with current background settings
            buf_final = io.BytesIO()
            if bg_transparency == 100:
                # Save as PNG with transparency
                final_image.save(buf_final, format="PNG")
            else:
                # Save as JPEG for colored backgrounds (smaller file size)
                final_image.convert("RGB").save(buf_final, format="PNG")
            buf_final.seek(0)
            
            st.download_button(
                "💾 Download Result",
                buf_final,
                file_name="visionextract_result.png",
                mime="image/png",
                use_container_width=True,
                help="Download with current background settings"
            )
        
        with col_dl2:
            # Always provide transparent version
            buf_transparent = io.BytesIO()
            st.session_state["result_masked"].save(buf_transparent, format="PNG")
            buf_transparent.seek(0)
            st.download_button(
                "🔍 Transparent PNG",
                buf_transparent,
                file_name="visionextract_transparent.png",
                mime="image/png",
                use_container_width=True,
                help="Always download with transparent background"
            )
        
        # Optional: Show probability heatmap
        if st.checkbox("Show AI Confidence Heatmap"):
            avg_map = st.session_state["avg_prob_map"]
            hm = (255 * (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min() + 1e-8)).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            hm_pil = Image.fromarray(cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB))
            st.image(hm_pil, caption="AI Confidence Heatmap (Red = High Confidence)", use_column_width=True)
    
    else:
        st.info("👆 Upload an image and click 'Generate Mask' to see results here!")

# Footer
st.markdown("---")
st.markdown("### 🚀 How It Works")
st.markdown("""
- **AI Segmentation**: Uses advanced DeepLabV3 model trained on thousands of images
- **Smart Processing**: Automatically detects and isolates the main subject
- **Quality Output**: Provides transparent PNG or custom background options
- **Easy Download**: One-click download for your processed images
""")

st.markdown("---")
st.markdown("Built with ❤️ using PyTorch & Streamlit — VisionExtract AI Pipeline")