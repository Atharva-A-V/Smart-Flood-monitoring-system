import os, io, json, warnings, hashlib, time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from rasterio.io import MemoryFile
import rasterio
from rasterio.warp import transform as rio_transform
from rasterio.crs import CRS
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu
from sentence_transformers import SentenceTransformer
import torch, open_clip, kagglehub
from openai import OpenAI
from geopy.geocoders import Nominatim

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)

# ---------- CONFIG ----------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
MODEL = "gpt-4o-mini"
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Flood Detection", layout="wide")
st.title("🌊 Flood Detection and Risk Analysis")

DEFAULT_NDMI_THRESHOLD = 0.4
DEFAULT_AWEI_THRESHOLD = 0.5
SMALL_OBJECT_THRESHOLD = 64
SMALL_HOLE_THRESHOLD = 64
CACHE_DIR = os.path.expanduser("~/.cache/flood_rag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- HELPERS ----------
def norm01(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if mx - mn < 1e-9 else (x - mn) / (mx - mn)

def _band(a, i):
    return a[i] if i < a.shape[0] else np.zeros_like(a[0])

def file_hash(path):
    h = hashlib.sha256()
    for root, _, files in os.walk(path):
        for f in sorted(files)[:30]:
            h.update(f.encode())
    return h.hexdigest()[:16]

def get_location_from_meta(meta):
    """Extract centroid lat/lon from GeoTIFF metadata and reverse geocode."""
    try:
        transform = meta.get("transform")
        crs = meta.get("crs")
        width, height = meta["width"], meta["height"]
        if transform is None or crs is None:
            return None, None, None

        center_x = transform[2] + transform[0] * width / 2
        center_y = transform[5] + transform[4] * height / 2

        lon, lat = rio_transform(CRS.from_user_input(crs), CRS.from_epsg(4326), [center_x], [center_y])
        lat, lon = lat[0], lon[0]

        geolocator = Nominatim(user_agent="flood_location_lookup", timeout=10)
        location = geolocator.reverse((lat, lon), language="en", exactly_one=True)
        address = location.address if location else "Unknown location"
        return lat, lon, address
    except Exception:
        return None, None, None

# ---------- MODELS ----------
@st.cache_resource
def init_models():
    txt = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return txt, (clip_model.to("cpu"), clip_pre)

txt_embed, (clip_model, clip_pre) = init_models()

# ---------- COSINE SAFE FIX ----------
def cosine(a, b):
    if b is None or len(b.shape) != 2 or b.shape[0] == 0:
        return np.zeros(1)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.shape[-1] != b.shape[-1]:
        min_dim = min(a.shape[-1], b.shape[-1])
        a = a[:min_dim]
        b = b[:, :min_dim]
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(b, a)

# ---------- RAG LOADING (FIXED) ----------
@st.cache_resource
def load_rag():
    # --- NO STREAMLIT ELEMENTS INSIDE A CACHED FUNCTION ---
    try:
        path = kagglehub.dataset_download("tallspecsguy/hls-flood-dataset")
        print(f"Dataset loaded: {path}") # Use print for console logging
    except Exception as e:
        print(f"⚠️ Could not load dataset: {e}") # Use print
        return [], np.zeros((0, 512))

    cache_id = file_hash(path)
    cache_file = os.path.join(CACHE_DIR, f"rag_cache_{cache_id}.npz")

    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        docs, embs = data["docs"].tolist(), data["embs"]
        print(f"⚡ Loaded cached embeddings ({len(embs)} items)") # Use print
        return docs, embs

    docs, embs = [], []
    print("🧠 Building RAG embeddings (first-time setup)...") # Use print
    t0 = time.time()
    for root, _, files in os.walk(path):
        for f in files:
            fpath = os.path.join(root, f)
            try:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    im = Image.open(fpath).convert("RGB")
                elif f.lower().endswith((".tif", ".tiff")):
                    with rasterio.open(fpath) as src:
                        arr = src.read()
                        if arr.shape[0] >= 3:
                            arr = np.transpose(arr[:3], (1, 2, 0))
                            im = Image.fromarray((norm01(arr) * 255).astype(np.uint8))
                        else:
                            continue
                else:
                    continue
                with torch.no_grad():
                    v = clip_model.encode_image(clip_pre(im).unsqueeze(0))
                    v = (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
                # Store both path and caption
                docs.append({"path": fpath, "caption": f"Flood image from {f}"})
                embs.append(v)
            except Exception:
                continue

    if len(embs) == 0:
        print("⚠️ No flood images found in dataset.") # Use print
        return [], np.zeros((0, 512))

    embs = np.stack(embs)
    np.savez(cache_file, docs=docs, embs=embs)
    print(f"✅ Cached {len(embs)} embeddings in {time.time() - t0:.1f}s") # Use print
    return docs, embs

# --- Call RAG loading early so it's ready ---
with st.spinner("Loading AI models and flood data..."):
    rag_docs, rag_embs = load_rag()

if len(rag_embs) == 0:
    st.warning("⚠️ RAG dataset could not be loaded. AI Reasoning may be limited.")

def multimodal_rag_query(q, img=None):
    if rag_embs is None or len(rag_embs.shape) != 2 or rag_embs.shape[0] == 0:
        return "No RAG context available.", []
    sims = np.zeros(rag_embs.shape[0])
    if q:
        t_emb = txt_embed.encode([q], normalize_embeddings=True)[0]
        sims += 0.6 * cosine(t_emb, rag_embs)
    if img is not None:
        try:
            if img.ndim == 3 and img.shape[0] == 3:
                im = Image.fromarray((np.transpose(img * 255, (1, 2, 0))).astype(np.uint8))
            else:
                im = Image.fromarray((img * 255).astype(np.uint8))
            with torch.no_grad():
                v = clip_model.encode_image(clip_pre(im).unsqueeze(0))
                v = (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
            sims += 0.4 * cosine(v, rag_embs)
        except Exception:
            pass
    if np.all(sims == 0):
        return "No relevant flood images found.", []
    
    top = np.argsort(-sims)[:3]
    
    # --- UPDATED ---
    # Return both the text captions AND the image paths
    top_docs = [rag_docs[i] for i in top if i < len(rag_docs)]
    context_str = "\n".join(doc["caption"] for doc in top_docs)
    image_paths = [doc["path"] for doc in top_docs]
    
    return context_str, image_paths

# ---------- IMAGE + GEO PROCESS ----------
def load_image_any(f):
    data = f.read()
    meta = {}
    try:
        with MemoryFile(data) as mem:
            with mem.open() as ds:
                arr, meta = ds.read().astype(np.float32), ds.meta
    except Exception:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.transpose(np.array(im), (2, 0, 1)).astype(np.float32)
        meta = {"transform": None, "crs": None, "width": arr.shape[2], "height": arr.shape[1]}
    for b in range(arr.shape[0]):
        v = arr[b]
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        arr[b] = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    return arr, meta

def compute_indices(a):
    R, G, B = _band(a, 0), _band(a, 1), _band(a, 2)
    NIR, SWIR1, SWIR2 = _band(a, 3), _band(a, 4), _band(a, 5)
    eps = 1e-6
    NDWI = (G - NIR) / (G + NIR + eps)
    MNDWI = (G - SWIR1) / (G + SWIR1 + eps)
    NDMI = (NIR - SWIR1) / (NIR + SWIR1 + eps)
    swir2_eff = SWIR2 if np.any(SWIR2) else SWIR1
    AWEI = 4 * (G - SWIR1) - (0.25 * NIR + 2.75 * swir2_eff)
    AWEI = 2 * norm01(AWEI) - 1
    RGB = np.stack([R, G, B], -1)
    return {"NDWI": NDWI, "MNDWI": MNDWI, "NDMI": NDMI, "AWEI": AWEI, "RGB": RGB}

def flood_mask(idx, thr, ndmi, awei, ndmi_thr, awei_thr):
    raw = idx > thr
    mask = raw & (ndmi < ndmi_thr) & (awei < awei_thr)
    mask = remove_small_objects(mask, SMALL_OBJECT_THRESHOLD)
    mask = remove_small_holes(mask, SMALL_HOLE_THRESHOLD)
    return mask.astype(np.uint8)

def overlay(rgb, m):
    o = rgb.copy()
    r = np.zeros_like(rgb)
    r[..., 0] = 1
    return np.clip(np.where(m[..., None].astype(bool), r * 0.9 + o * 0.4, o), 0, 1)

@st.cache_data(show_spinner="Analyzing flood image...")
def run_analysis(uploaded_file, ndmi_thr, awei_thr):
    uploaded_file.seek(0)
    arr, meta = load_image_any(uploaded_file)
    lat, lon, address = get_location_from_meta(meta)
    idxs = compute_indices(arr)
    thr = float(2 * ((threshold_otsu(idxs["NDWI"]) - idxs["NDWI"].min()) /
                     (idxs["NDWI"].max() - idxs["NDWI"].min() + 1e-9)) - 1)
    mask = flood_mask(idxs["NDWI"], thr, idxs["NDMI"], idxs["AWEI"], ndmi_thr, awei_thr)
    overlay_img = overlay(idxs["RGB"], mask)
    flood_pct = 100 * (mask.sum() / (mask.size + 1e-9))
    risk = float(np.tanh(3 * (mask.sum() / (mask.size + 1e-9))))
    
    # --- UPDATED ---
    rag_ctx, rag_img_paths = multimodal_rag_query("Flood pattern similarity", arr)
    
    # Return the new image paths list
    return arr, idxs, thr, mask, overlay_img, flood_pct, risk, rag_ctx, rag_img_paths, lat, lon, address

# ---------- UI ----------
with st.sidebar:
    st.header("Analysis Settings")
    ndmi_thr = st.slider("NDMI Threshold (Vegetation Filter)", -1.0, 1.0, DEFAULT_NDMI_THRESHOLD, 0.05)
    awei_thr = st.slider("AWEI Threshold (Shadow Filter)", -1.0, 1.0, DEFAULT_AWEI_THRESHOLD, 0.05)
    
    st.divider() 
    
    st.header("Upload Image")
    uploaded = st.file_uploader("Upload Satellite Image (GeoTIFF, JPG, PNG)", label_visibility="collapsed")

# --- DEFINE TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Analysis Summary", "Export Report", "AI Reasoning Chat", "Methodology", "Feature Analytics"
])

# --- Tab 1 ---
with tab1:
    if not uploaded:
        st.info("Please upload an image via the sidebar to begin flood analysis.")
    else:
        # --- UPDATED ---
        arr, idxs, thr, mask, overlay_img, flood_pct, risk, rag_ctx, rag_img_paths, lat, lon, address = run_analysis(uploaded, ndmi_thr, awei_thr)

        st.subheader("Detected Flood Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Flooded Area", f"{flood_pct:.2f}%")
        col2.metric("Risk Score", f"{risk:.2f}")
        col3.metric("NDWI Threshold (Otsu)", f"{thr:.2f}")

        if lat and lon:
            st.markdown(f"**📍 Location:** {address or 'Unknown'}")
            st.markdown(f"**🌐 Coordinates:** Latitude `{lat:.5f}`, Longitude `{lon:.5f}`")
        else:
            st.warning("⚠️ Could not extract geolocation from image metadata.")

        st.subheader("Spectral Indices Visualization")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes[0, 0].imshow(idxs["RGB"]); axes[0, 0].set_title("RGB (True Color)"); axes[0, 0].axis("off")
        axes[0, 1].imshow(idxs["NDWI"], cmap="Blues", vmin=-1, vmax=1); axes[0, 1].set_title("NDWI (Water)"); axes[0, 1].axis("off")
        axes[0, 2].imshow(idxs["MNDWI"], cmap="Greens", vmin=-1, vmax=1); axes[0, 2].set_title("MNDWI (Water)"); axes[0, 2].axis("off")
        axes[1, 0].imshow(idxs["NDMI"], cmap="YlGn", vmin=-1, vmax=1); axes[1, 0].set_title("NDMI (Moisture)"); axes[1, 0].axis("off")
        axes[1, 1].imshow(idxs["AWEI"], cmap="Purples", vmin=-1, vmax=1); axes[1, 1].set_title("AWEI (Water)"); axes[1, 1].axis("off")
        axes[1, 2].imshow(overlay_img); axes[1, 2].set_title("Flood Detection Overlay"); axes[1, 2].axis("off")
        plt.tight_layout()
        st.pyplot(fig)

# --- Tab 2 ---
with tab2:
    st.subheader("Export Report")
    if not uploaded:
        st.info("Upload an image via the sidebar to generate a report.")
    else:
        # --- UPDATED ---
        arr, idxs, thr, mask, overlay_img, flood_pct, risk, rag_ctx, rag_img_paths, lat, lon, address = run_analysis(uploaded, ndmi_thr, awei_thr)
        
        report_data = {
            "location": {
                "latitude": lat,
                "longitude": lon,
                "address": address
            },
            "analysis_parameters": {
                "ndmi_threshold_setting": ndmi_thr,
                "awei_threshold_setting": awei_thr,
                "otsu_ndwi_threshold_calculated": thr
            },
            "results": {
                "flooded_area_percent": flood_pct,
                "risk_score": risk
            },
            # --- UPDATED ---
            "rag_context": {
                "captions": rag_ctx.split('\n'),
                "image_paths": rag_img_paths
            }
        }
        st.json(report_data)
        st.download_button(
            "Download JSON Report",
            data=json.dumps(report_data, indent=2),
            file_name="flood_analysis_report.json",
            mime="application/json"
        )

# --- Tab 3 (CHATBOT) ---
with tab3:
    st.subheader("AI-Powered Reasoning Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_file" not in st.session_state:
        st.session_state.current_file = None

    if not uploaded:
        st.info("Upload an image via the sidebar to start chatting about the analysis.")
    else:
        # --- UPDATED ---
        # Get data for context, including the new rag_img_paths
        arr, idxs, thr, mask, overlay_img, flood_pct, risk, rag_ctx, rag_img_paths, lat, lon, address = run_analysis(uploaded, ndmi_thr, awei_thr)
        
        # Reset chat if file changes
        if st.session_state.current_file != uploaded.name:
            st.session_state.messages = []
            st.session_state.current_file = uploaded.name

        if not client:
            st.warning("⚠️ OpenAI API key not found. AI reasoning is unavailable.")
        else:
            # --- NEW: Display RAG Images ---
            with st.expander("See RAG Context Images (Top 3 Matches)"):
                if not rag_img_paths:
                    st.write("No similar images found in the RAG database.")
                else:
                    captions = rag_ctx.split('\n')
                    for i, (img_path, caption) in enumerate(zip(rag_img_paths, captions)):
                        try:
                            image = Image.open(img_path)
                            st.image(image, caption=f"Match {i+1}: {caption}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not load RAG image: {img_path}. Error: {e}")
            
            # --- End of New Section ---

            # Define the system prompt with all context
            system_prompt = f"""
            You are a concise flood analysis expert. Your job is to interpret numerical flood data for a professional audience.
            You are in a chat with a user analyzing a specific satellite image.
            
            Here is the complete context for the *current* image:
            
            Location Data:
            - Address: {address or 'N/A'}
            - Coordinates: ({lat:.5f}, {lon:.5f})

            Analysis Data:
            - Flooded area: {flood_pct:.2f}%
            - Risk score: {risk:.2f} (from 0 to 1)
            - Calculated NDWI Threshold: {thr:.2f}

            Historical Context (from similar images in the database):
            - {rag_ctx}
            
            You have already shown the user the actual images corresponding to this historical context.
            Use all this context to answer user questions. Be direct and professional.
            """
            
            # Auto-generate first message if chat is new
            if not st.session_state.messages:
                initial_user_prompt = "Based on all the context, provide a concise, expert summary of the flood situation."
                try:
                    with st.spinner("AI is analyzing the results..."):
                        r = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": initial_user_prompt}
                            ],
                            max_tokens=250
                        )
                        summary = r.choices[0].message.content.strip()
                        st.session_state.messages.append({"role": "assistant", "content": summary})
                except Exception as e:
                    st.error(f"An error occurred while contacting the AI model: {e}")

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Get user input
            if prompt := st.chat_input("Ask about this flood analysis..."):
                # Add user message to state and display
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Construct the full message list
                            messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                            
                            r = client.chat.completions.create(
                                model=MODEL,
                                messages=messages_to_send,
                                max_tokens=300
                            )
                            response = r.choices[0].message.content.strip()
                            st.markdown(response)
                            # Add assistant response to state
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

# --- Tab 4 ---
with tab4:
    st.subheader("Spectral Index Formulas")
    st.markdown("""
    The flood detection model relies on several well-established spectral indices to
    differentiate water from other land features.
    
    ---
    
    **NDWI (Normalized Difference Water Index)**
    
    Highlights open water bodies. It is effective at separating water from land,
    but can sometimes confuse water with built-up areas.
    $$NDWI = \\frac{Green - NIR}{Green + NIR}$$
    
    ---
    
    **MNDWI (Modified NDWI)**
    
    Enhances open water features while suppressing noise from built-up land,
    making it more reliable than NDWI in urban or semi-urban environments.
    $$MNDWI = \\frac{Green - SWIR1}{Green + SWIR1}$$
    
    ---
    
    **NDMI (Normalized Difference Moisture Index)**
    
    Measures vegetation water content. It is used here as a filter: areas with
    high moisture (high NDMI) are likely healthy vegetation, not open floodwater.
    $$NDMI = \\frac{NIR - SWIR1}{NIR + SWIR1}$$
    
    ---
    
    **AWEI (Automated Water Extraction Index)**
    
    Designed to improve water extraction in areas with shadow and dark surfaces
    (e.g., building shadows, dark soil), which are often misclassified as water.
    $$AWEI = 4 \cdot (Green - SWIR1) - (0.25 \cdot NIR + 2.75 \cdot SWIR2)$$
    """)

# --- Tab 5 ---
with tab5:
    st.subheader("Spectral Feature Analytics")
    if not uploaded:
        st.info("Upload an image via the sidebar to view feature analytics.")
    else:
        # --- UPDATED ---
        arr, idxs, thr, mask, overlay_img, flood_pct, risk, rag_ctx, rag_img_paths, lat, lon, address = run_analysis(uploaded, ndmi_thr, awei_thr)

        flat = {"NDWI": idxs["NDWI"].ravel(), "MNDWI": idxs["MNDWI"].ravel(),
                "NDMI": idxs["NDMI"].ravel(), "AWEI": idxs["AWEI"].ravel()}
        df = pd.DataFrame(flat).dropna()
        
        col1, col2 = st.columns(2)
        thresholds = np.linspace(-0.5, 0.5, 40)
        flood_fracs = [((idxs["NDWI"] > t) & (idxs["NDMI"] < DEFAULT_NDMI_THRESHOLD)
                        & (idxs["AWEI"] < DEFAULT_AWEI_THRESHOLD)).sum() / idxs["NDWI"].size * 100 for t in thresholds]
        
        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(df["NDWI"], bins=50, ax=ax1, kde=True, color="skyblue")
            ax1.axvline(thr, color="red", linestyle="--", label=f"Otsu Threshold ({thr:.2f})")
            ax1.legend()
            ax1.set_title("NDWI Distribution and Otsu Threshold")
            ax1.set_xlabel("NDWI Value")
            ax1.set_ylabel("Pixel Count")
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots()
            ax2.plot(thresholds, flood_fracs, "-o", color="blue")
            ax2.axvline(thr, color="red", linestyle="--", label="Selected (Otsu) Threshold")
            ax2.set_xlabel("NDWI Threshold")
            ax2.set_ylabel("Calculated Flood Area (%)")
            ax2.legend()
            ax2.set_title("Flood Area Sensitivity to NDWI Threshold")
            st.pyplot(fig2)

        st.markdown(
            """
            ### Plot Interpretations
            
            * **NDWI Distribution (Left):** This histogram shows the frequency of all NDWI pixel values in the image. 
                Often, you will see two main "humps": one for non-water surfaces (like land and buildings) and another 
                for water. The red dashed line is the **Otsu Threshold**, which is the value automatically 
                calculated to best separate these two groups.
                
            * **Flood Area Sensitivity (Right):** This plot shows how the *total* calculated flooded area would change 
                if we manually picked a different NDWI threshold. A steep slope around the red line means the 
                final flood percentage is highly sensitive to the threshold. A flatter slope suggests a more stable
                and clear-cut separation between water and non-water areas.
            """
        )

        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            fig3, ax3 = plt.subplots()
            sample = df.sample(n=4000) if len(df) > 4000 else df
            sns.scatterplot(data=sample, x="MNDWI", y="NDMI", s=10, alpha=0.3, ax=ax3, edgecolor=None)
            ax3.set_xlabel("MNDWI (Water)")
            ax3.set_ylabel("NDMI (Moisture)")
            ax3.set_title("MNDWI vs. NDMI Pixel Correlation")
            st.pyplot(fig3)
            
        with col4:
            fig4, ax4 = plt.subplots()
            sns.heatmap(df.corr(), annot=True, cmap="vlag", ax=ax4, fmt=".2f", center=0)
            ax4.set_title("Spectral Index Correlation Heatmap")
            st.pyplot(fig4)
            
        st.markdown(
            """
            ### Plot Interpretations
            
            * **MNDWI vs. NDMI Scatter (Left):** This plot helps visualize the relationship between "water-ness" (MNDWI)
                and "vegetation-ness" (NDMI) for thousands of sample pixels. Pure water pixels should appear in the 
                top-left quadrant (high MNDWI, low NDMI). Vegetated pixels would be in the top-right (high NDMI). 
                This helps confirm that our filters (like `NDMI < threshold`) are effective.
                
            * **Correlation Heatmap (Right):** This matrix shows how strongly each index relates to the others. 
                A value near **1.0** (dark red) means the two indices act very similarly (e.g., NDWI and MNDWI, 
                as both detect water). A value near **-1.0** (dark blue) would mean they are opposites. 
                This is a high-level check to ensure the spectral data is behaving as expected.
            """
)