from pyscript import document, display, when
import numpy as np
from PIL import Image, ImageFilter
import io

# --- CORE LOGIC (Copied from halation.py) ---

def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)

def _gaussian_blur_np(img01: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return img01
    im = Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8), mode="RGB")
    im = im.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(im).astype(np.float32) / 255.0

def _apply_chromatic_aberration(img_linear: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img_linear
        
    h, w, c = img_linear.shape
    
    # Simple radial aberration scaling
    # We will scale Red channel up and Blue channel down, keeping Green as anchor
    
    # Convert to PIL for easy resizing
    img_srgb_uint8 = (np.clip(_linear_to_srgb(img_linear), 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_srgb_uint8)
    r, g, b = pil_img.split()
    
    # Calculate scale factors
    # amount is percentage shift at edge roughly
    scale_r = 1.0 + amount
    scale_b = 1.0 - amount
    
    # Resize Red
    new_w_r = int(w * scale_r)
    new_h_r = int(h * scale_r)
    r_resized = r.resize((new_w_r, new_h_r), Image.BICUBIC)
    # Crop center of Red
    left = (new_w_r - w) // 2
    top = (new_h_r - h) // 2
    r_final = r_resized.crop((left, top, left + w, top + h))
    
    # Resize Blue
    new_w_b = int(w * scale_b)
    new_h_b = int(h * scale_b)
    b_resized = b.resize((new_w_b, new_h_b), Image.BICUBIC)
    # Paste centered into black canvas of original size
    b_final = Image.new("L", (w, h))
    left = (w - new_w_b) // 2
    top = (h - new_h_b) // 2
    b_final.paste(b_resized, (left, top))
    
    # Merge back
    final_pil = Image.merge("RGB", (r_final, g, b_final))
    return _srgb_to_linear(np.asarray(final_pil).astype(np.float32) / 255.0)

def apply_effects(
    img: Image.Image,
    halation_strength: float = 1.3,
    halation_threshold: float = 0.75,
    halation_softness: float = 2.0,
    halation_blur_radius: float = 60.0,
    bloom_strength: float = 0.8,
    bloom_radius: float = 120.0,
    bloom_tint: tuple[float, float, float] = (1.0, 0.85, 0.6),
    streak_strength: float = 0.35,
    streak_threshold: float = 0.85,
    streak_stretch: float = 50.0,
    aberration_amount: float = 0.001,
    grain_amount: float = 0.012,
    grain_size: float = 1.0,
    grain_color: float = 0.20,
    seed: int | None = None,
) -> Image.Image:
    
    rng = np.random.default_rng(seed)

    img = img.convert("RGB")
    srgb = np.asarray(img).astype(np.float32) / 255.0
    lin = _srgb_to_linear(srgb)

    # Luminance (linear)
    y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    
    # Brightness (Max of RGB)
    bright = np.max(lin, axis=2)

    # --- Halation ---
    if halation_strength > 0:
        t = float(halation_threshold)
        m = np.clip((bright - t) / max(1e-6, (1.0 - t)), 0.0, 1.0)
        m = m ** float(halation_softness)
        bloom_src = lin * m[..., None]
        bloom_blur_srgb = _gaussian_blur_np(_linear_to_srgb(bloom_src), halation_blur_radius)
        bloom_blur = _srgb_to_linear(bloom_blur_srgb)
        
        tint = np.array(halation_tint, dtype=np.float32)
        tint = tint / max(1e-6, float(np.max(tint)))
        bloom_luma = (bloom_blur[..., 0] + bloom_blur[..., 1] + bloom_blur[..., 2]) / 3.0
        halation_layer = (bloom_blur * 0.35 + (bloom_luma[..., None] * tint[None, None, :]) * 0.65)
        
        lin = lin + halation_layer * float(halation_strength)
    
    # --- Bloom (Golden Foggy Overlay) ---
    if bloom_strength > 0:
        bloom_t = 0.6
        bloom_mask = np.clip((bright - bloom_t) / max(1e-6, (1.0 - bloom_t)), 0.0, 1.0)
        bloom_src = lin * bloom_mask[..., None]
        bloom_blur_srgb = _gaussian_blur_np(_linear_to_srgb(bloom_src), bloom_radius)
        bloom_blur = _srgb_to_linear(bloom_blur_srgb)
        
        b_tint = np.array(bloom_tint, dtype=np.float32)
        b_tint = b_tint / max(1e-6, float(np.max(b_tint)))
        
        blur_luma = (bloom_blur[..., 0] + bloom_blur[..., 1] + bloom_blur[..., 2]) / 3.0
        
        # 100% Tinted Overlay
        tinted_bloom = (blur_luma[..., None] * b_tint[None, None, :])

        lin = lin + tinted_bloom * float(bloom_strength)

    # --- Anamorphic Streaks ---
    if streak_strength > 0:
        streak_t = float(streak_threshold)
        streak_mask = np.clip((bright - streak_t) / max(1e-6, (1.0 - streak_t)), 0.0, 1.0)
        streak_mask = streak_mask ** 4.0
        streak_src = lin * streak_mask[..., None]
        
        h, w, c = streak_src.shape
        w_small = max(1, int(w / float(streak_stretch)))
        
        streak_src_pil = Image.fromarray((np.clip(_linear_to_srgb(streak_src), 0, 1) * 255).astype(np.uint8))
        streak_shrunk = streak_src_pil.resize((w_small, h), resample=Image.BILINEAR)
        streak_shrunk = streak_shrunk.filter(ImageFilter.GaussianBlur(radius=1.0))
        streak_stretched = streak_shrunk.resize((w, h), resample=Image.BICUBIC)
        streak_final = _srgb_to_linear(np.asarray(streak_stretched).astype(np.float32) / 255.0)
        
        lin = lin + streak_final * float(streak_strength)

    # --- Chromatic Aberration ---
    if aberration_amount > 0:
        lin = _apply_chromatic_aberration(lin, aberration_amount)

    # --- Grain ---
    h, w = y.shape
    base_noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    if grain_size > 1.0:
        n01 = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min() + 1e-6)
        n_img = Image.fromarray((n01 * 255).astype(np.uint8), mode="L")
        n_img = n_img.filter(ImageFilter.GaussianBlur(radius=float(grain_size)))
        n = np.asarray(n_img).astype(np.float32) / 255.0
        base_noise = (n - 0.5) * 2.0
    
    y_final = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    weight = (1.0 - np.clip(y_final, 0.0, 1.0)) ** 0.6
    grain = base_noise * weight * float(grain_amount)
    
    color_noise = rng.normal(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
    color_noise = (color_noise - color_noise.mean(axis=2, keepdims=True))
    grain_rgb = grain[..., None] + color_noise * (float(grain_amount) * float(grain_color) * 0.35)

    out = lin + grain_rgb
    out = np.clip(out, 0.0, 1.0)
    out_srgb = (_linear_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out_srgb, mode="RGB")

# --- UI INTERACTION ---

uploaded_image = None
processed_image = None

@when("change", "#image-upload")
def handle_upload(event):
    global uploaded_image
    file_input = document.getElementById("image-upload")
    file_list = file_input.files
    
    if len(file_list) > 0:
        file = file_list.item(0)
        
        # Read file
        async def process_file(f):
            array_buf = await f.arrayBuffer()
            bytes_data = array_buf.to_bytes()
            img = Image.open(io.BytesIO(bytes_data))
            
            # Update global state
            global uploaded_image
            uploaded_image = img
            
            # Show preview (source)
            # Create object URL for preview
            # For simplicity in PyScript, we can convert back to base64 or similar, 
            # but cleaner to just let the user process it.
            # Let's auto-process to show result? 
            # Or at least show "Ready to process". For now, user clicks Process.
            document.getElementById("image-container").classList.remove("empty-state")
            document.querySelector(".empty-state p").innerText = "Image Loaded. Click 'Apply Effects'."
            
            # Display source (optional, maybe we just show result)
            # display_image(img, "source-img")

        # PyScript async handler
        import asyncio
        asyncio.ensure_future(process_file(file))

@when("click", "#process-btn")
def process_click(event):
    global uploaded_image, processed_image
    if uploaded_image is None:
        return
    
    document.getElementById("loading").style.display = "block"
    document.getElementById("result-img").style.display = "none"
    
    # Needs to be async to allow UI update
    import asyncio
    asyncio.ensure_future(run_processing())

async def run_processing():
    global uploaded_image, processed_image
    
    # Get values from UI
    h_str = float(document.getElementById("halation_strength").value)
    h_rad = float(document.getElementById("halation_blur_radius").value)
    h_th = float(document.getElementById("halation_threshold").value)
    
    b_str = float(document.getElementById("bloom_strength").value)
    b_rad = float(document.getElementById("bloom_radius").value)
    
    s_str = float(document.getElementById("streak_strength").value)
    
    g_amt = float(document.getElementById("grain_amount").value)
    a_amt = float(document.getElementById("aberration_amount").value)
    
    # Process
    # Resize for preview speed? For now, process full res.
    # If image is HUGE, maybe downscale.
    proc_img = uploaded_image.copy()
    
    # MAX WIDTH for performance in browser
    MAX_WIDTH = 2000
    if proc_img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / proc_img.width
        proc_img = proc_img.resize((MAX_WIDTH, int(proc_img.height * ratio)))

    result = apply_effects(
        proc_img,
        halation_strength=h_str,
        halation_threshold=h_th,
        halation_blur_radius=h_rad,
        bloom_strength=b_str,
        bloom_radius=b_rad,
        streak_strength=s_str,
        grain_amount=g_amt,
        aberration_amount=a_amt
    )
    
    processed_image = result
    
    # Display result
    # Convert PIL to Bytes
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    byte_data = buf.getvalue()
    
    # Create Object URL in JS land? 
    # Or base64. Base64 is easier in pure Python-to-DOM without extra JS bridge.
    import base64
    b64 = base64.b64encode(byte_data).decode('utf-8')
    src = f"data:image/png;base64,{b64}"
    
    img_el = document.getElementById("result-img")
    img_el.src = src
    img_el.style.display = "block"
    
    document.getElementById("loading").style.display = "none"
    
    # Enable Download
    dl_btn = document.getElementById("download-btn")
    dl_btn.disabled = False
    
    # Setup download click (one-time listener or just direct property set?)
    # PyScript interaction with existing JS is sometimes tricky.
    # Let's use a JS function call to handle download if we can, or pure python.
    # Pure python:
    from pyscript import window
    window.download_blob = src # Pass data to global JS to handle download?
    # Or just add event listener here
    
@when("click", "#download-btn")
def download_click(event):
    global processed_image
    if processed_image is None:
        return
        
    buf = io.BytesIO()
    processed_image.save(buf, format="PNG")
    byte_data = buf.getvalue()
    
    import base64
    b64 = base64.b64encode(byte_data).decode('utf-8')
    src = f"data:image/png;base64,{b64}"
    
    # Create dummy link
    from js import document as js_doc
    link = js_doc.createElement("a")
    link.href = src
    link.download = "processed_image.png"
    js_doc.body.appendChild(link)
    link.click()
    js_doc.body.removeChild(link)
