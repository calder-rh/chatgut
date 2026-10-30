import base64
import io
import json
import os
import html
import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

# Set page layout with sidebar collapsed by default
st.set_page_config(
    page_title="Chatbot", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

def d2xy_hilbert(n, d):
    """Maps a 1D index d to a 2D (x, y) coordinate along an n x n classical Hilbert curve."""
    rx = ry = 0
    t = d
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def hybrid_hilbert_serpentine_indices(width=19, height=16):
    """
    Generates a hybrid spatial curve for a 16x19 grid:
    - Left 16x16: True 2D Hilbert Curve (indices 0..255)
    - Right 16x3 extension: Serpentine/Snake Curve (indices 256..303)
    """
    coords = []
    
    # 1. Map left 16x16 square via true Hilbert curve
    for d in range(16 * 16):
        x, y = d2xy_hilbert(16, d)
        coords.append((x, y))
        
    # 2. Map remaining 16x3 right extension via Serpentine curve
    for y in range(16):
        x_range = range(16, 19) if y % 2 == 0 else range(18, 15, -1)
        for x in x_range:
            coords.append((x, y))
            
    # Convert (x, y) coordinates into flat 1D array index: [Y * Width + X]
    return [y * width + x for x, y in coords]

@st.cache_resource
def load_model_and_embeddings():
    """Loads model and maps embeddings to a 16x19 hybrid Hilbert/Serpentine grid."""
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32, 
        device_map="auto"
    )
    
    # Get weight matrix: shape [vocab_size, hidden_dim]
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    hidden_dim = embeddings.shape[1] # 896
    
    # Target Grid Dimensions: 19 wide x 16 high (304 cells)
    width = 19
    height = 16
    
    # 896 dimensions / 3 RGB channels = 299 pixels needed out of 304 grid cells
    num_pixels = int(np.ceil(hidden_dim / 3.0))
    
    # Compute 1D hierarchical feature clustering order
    vocab_sample = embeddings[:5000].T
    distances = pdist(vocab_sample, metric='cosine')
    linkage_matrix = linkage(distances, method='ward')
    cluster_order = leaves_list(linkage_matrix)
    
    # Get 16x19 Hybrid curve index permutation
    hybrid_indices = hybrid_hilbert_serpentine_indices(width, height)
    
    vocab_mean = float(np.mean(embeddings))
    vocab_std = float(np.std(embeddings))
    
    return tokenizer, model, embeddings, cluster_order, hybrid_indices, width, height, num_pixels, vocab_mean, vocab_std

tokenizer, model, raw_embeddings, cluster_order, hybrid_indices, W, H, NUM_PIXELS, MEAN, STD = load_model_and_embeddings()

# Sidebar Control
st.sidebar.title("Settings")
STABILITY_CHAR_WINDOW = st.sidebar.slider(
    "Stability Character Lookback", 
    min_value=5, 
    max_value=50, 
    value=15,
    help="Number of trailing characters a token must remain unchanged across prefix evaluations before converting into an image."
)

MAX_GEN_TOKENS = st.sidebar.slider(
    "Max Response Tokens", 
    min_value=64, 
    max_value=1024, 
    value=512,
    step=64,
    help="Controls the max token generation limit for assistant responses."
)

@st.cache_data
def token_to_image_base64(token_id: int, scale_factor: int = 3) -> str:
    """Converts token vector into a 16x19 4-channel RGBA image scaled down to 57x48."""
    token_id = min(max(0, token_id), len(raw_embeddings) - 1)
    vector = raw_embeddings[token_id][cluster_order]
    
    # Normalize vector to 0..255 channel range
    normalized = (vector - MEAN) / (3 * STD) + 0.5
    clipped = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    
    # Pad dimension vector to split evenly into 3 RGB channels
    pad_len = (3 - (len(clipped) % 3)) % 3
    if pad_len > 0:
        clipped = np.pad(clipped, (0, pad_len), mode='edge')
        
    rgb_pixels = clipped.reshape(-1, 3)
    
    # Add Alpha Channel (255 = fully opaque)
    alphas = np.full((len(rgb_pixels), 1), 255, dtype=np.uint8)
    rgba_pixels = np.hstack([rgb_pixels, alphas])
    
    # Pad remaining 5 grid cells with transparent pixels [0, 0, 0, 0]
    total_grid_pixels = W * H
    if len(rgba_pixels) < total_grid_pixels:
        transparent_padding = np.zeros((total_grid_pixels - len(rgba_pixels), 4), dtype=np.uint8)
        rgba_pixels = np.vstack([rgba_pixels, transparent_padding])
        
    # Reorder pixels along hybrid Hilbert + Serpentine curve
    spatial_pixels = rgba_pixels[hybrid_indices]
    img_array = spatial_pixels.reshape((H, W, 4))
    
    img = Image.fromarray(img_array, mode="RGBA")
    
    # Upscale 19x16 -> 57x48
    new_width = W * scale_factor   # 57px
    new_height = H * scale_factor  # 48px
    upscaled_img = img.resize((new_width, new_height), resample=Image.NEAREST)
    
    buffered = io.BytesIO()
    upscaled_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_token_ids(text: str) -> list:
    """Extracts exact token IDs from text safely using Qwen tokenizer."""
    if not isinstance(text, str):
        return []
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    elif isinstance(encoded, torch.Tensor):
        return encoded.tolist()
    return list(encoded)

def calculate_scaled_fontsize(text_str: str, max_width_px: int = 55) -> int:
    """Calculates scaled font size in px to fit cleanly within compact container width."""
    char_len = max(1, len(text_str.strip()))
    approx_font_size = int(max_width_px / (char_len * 0.65))
    return int(np.clip(approx_font_size, 9, 13))

def clean_token_text(raw_text: str) -> str:
    """Safely escapes HTML and disables Streamlit markdown emoji/icon parsing."""
    escaped = html.escape(raw_text)
    return escaped.replace(":", "&#58;")

def render_tokens_html(text: str) -> str:
    """Converts text using exact HF tokenization into non-wrapping word blocks for chat bubbles."""
    if not isinstance(text, str) or not text.strip():
        return ""
        
    token_ids = get_token_ids(text)
    if not token_ids:
        return ""
        
    words_html = []
    current_word_spans = []

    for idx, token_id in enumerate(token_ids):
        raw_token_str = tokenizer.decode([token_id])
        clean_text = clean_token_text(raw_token_str)
        hover_display_str = clean_text.strip()
        b64_img = token_to_image_base64(token_id)
        scaled_font_size = calculate_scaled_fontsize(hover_display_str)
        
        has_leading_space = raw_token_str.startswith(" ") or raw_token_str.startswith("Ġ")
        
        if has_leading_space and idx > 0 and current_word_spans:
            words_html.append(f'<span class="word-wrapper" style="margin-left: 10px;">{"".join(current_word_spans)}</span>')
            current_word_spans = []

        span_html = f'<span class="token-container" style="width:57px;height:48px;"><img src="data:image/png;base64,{b64_img}" class="token-img" onerror="this.style.display=\'none\'" /><span class="token-hover-text" style="font-size:{scaled_font_size}px;">{hover_display_str}</span></span>'
        current_word_spans.append(span_html)

    if current_word_spans:
        margin_left = 10 if (len(words_html) > 0) else 0
        words_html.append(f'<span class="word-wrapper" style="margin-left: {margin_left}px;">{"".join(current_word_spans)}</span>')

    return f'<div class="chat-bubble-wrapper">{"".join(words_html)}</div>'

# Strictly targeted CSS rules for IBM Plex Mono
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&display=swap');

/* Apply IBM Plex Mono to chat text content while explicitly preserving avatar icons */
.centered-title, 
#single-editor, 
#single-editor *, 
.plain-token-text, 
.token-hover-text, 
div[data-testid="stChatMessageContent"] * {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Ensure Chat Avatars use native Streamlit icon font */
div[data-testid="stChatMessageAvatar"] * {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons', sans-serif !important;
}

/* Centered lightweight title */
.centered-title {
    text-align: center !important;
    font-size: 1.8rem !important;
    font-weight: 300 !important;
    letter-spacing: -0.5px;
    margin-top: 1rem;
    margin-bottom: 2rem;
    color: #ffffff;
}

/* Page layout spacing to prevent input box from covering chat history */
.stApp {
    padding-bottom: 300px !important;
}

/* Enforce fixed bottom position on the custom component parent container */
div[data-testid="stCustomComponentV1"], 
div[data-testid="stElementContainer"]:has(div[data-testid="stCustomComponentV1"]) {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    z-index: 999999 !important;
    background: #0e1117 !important;
    box-sizing: border-box !important;
}

/* Inline container for full words to prevent wrapping inside a single word */
.word-wrapper {
    display: inline-block !important;
    white-space: nowrap !important;
    vertical-align: middle !important;
    margin-top: 6px !important;
    margin-bottom: 6px !important;
}

.token-container {
    position: relative;
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    vertical-align: middle !important;
    margin-left: 1px;
    margin-right: 1px;
    cursor: default !important;
    line-height: 1;
    background: transparent;
    border-radius: 0px !important;
    overflow: hidden;
    user-select: none !important;
    width: 57px;
    height: 48px;
}

/* Morph animation ONLY runs once when a token first transitions from plain text to image */
.token-container.just-converted {
    animation: morphToken 0.25s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

@keyframes morphToken {
    from {
        opacity: 0.2;
        transform: scale(0.85);
        width: var(--initial-width, 30px);
    }
    to {
        opacity: 1;
        transform: scale(1);
        width: 57px;
    }
}

.token-img {
    width: 100% !important;
    height: 100% !important;
    vertical-align: middle !important;
    display: block !important;
    object-fit: fill !important;
    image-rendering: pixelated !important;
    image-rendering: -moz-crisp-edges !important;
    image-rendering: crisp-edges !important;
    border: none !important;
    box-sizing: border-box;
    border-radius: 0px !important;
    opacity: 1;
    transition: opacity 0.4s ease-in-out 4s;
    user-select: none !important;
    pointer-events: none !important;
}

.token-hover-text {
    display: flex !important;
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: #161b22;
    color: #e6edf3;
    font-weight: 400;
    white-space: pre-wrap;
    word-break: break-all;
    border-radius: 0px !important;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    border: 1px solid #30363d !important;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.4s ease-in-out 4s;
    overflow: hidden;
    padding: 1px;
    text-align: center;
}

.token-container:hover .token-img {
    opacity: 0 !important;
    transition: opacity 0s ease-in-out 0s !important;
}

.token-container:hover .token-hover-text {
    opacity: 1 !important;
    transition: opacity 0s ease-in-out 0s !important;
}

.plain-token-text {
    font-size: 14px;
    white-space: pre;
    background: rgba(255, 255, 255, 0.15);
    padding: 2px 5px;
    height: 48px;
    line-height: 44px;
    border-radius: 0px !important;
    margin-left: 1px;
    margin-right: 1px;
    display: inline-block;
    vertical-align: middle;
    box-sizing: border-box;
    cursor: default !important;
}

/* Active cursor container at end of prompt */
#active-text-node {
    outline: none;
    display: inline-block;
    min-width: 2px;
    height: 48px;
    line-height: 44px;
    vertical-align: middle;
    white-space: pre;
    color: #e6edf3;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown('<h1 class="centered-title">What can I help you with?</h1>', unsafe_allow_html=True)

# Pre-render common token dictionary & text lookup map
@st.cache_data
def get_common_vocab_data():
    vocab_map = {}
    for i in range(min(5000, tokenizer.vocab_size)):
        decoded_str = tokenizer.decode([i])
        b64_img = token_to_image_base64(i)
        vocab_map[decoded_str] = {"id": i, "b64": b64_img}
    return json.dumps(vocab_map)

VOCAB_MAP_JSON = get_common_vocab_data()
DEFAULT_IMAGE_B64 = token_to_image_base64(0)

# Build component folder cleanly
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "component_src")
os.makedirs(BUILD_DIR, exist_ok=True)

HTML_CONTENT = f"""
<!DOCTYPE html>
<html>
<head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&display=swap" rel="stylesheet">
{GLOBAL_CSS}
<style>
    body {{
        margin: 0;
        background: #0e1117;
        color: white;
        overflow: hidden;
    }}
    #editor-container {{
        width: 100%;
        background: #0e1117;
        padding: 10px 20px;
        box-sizing: border-box;
        border-top: 1px solid #30363d;
        position: relative;
    }}
    #single-editor {{
        min-height: 75px;
        max-height: 240px;
        background: #161b22;
        border: 1px solid #1f6beb;
        border-radius: 12px;
        padding: 10px 50px 10px 12px;
        color: #e6edf3;
        outline: none;
        font-size: 14px;
        line-height: 4.6;
        white-space: pre-wrap;
        word-break: break-word;
        overflow-y: auto;
        cursor: text;
        box-shadow: 0 0 0 1px rgba(31, 107, 235, 0.2);
    }}
    #single-editor:empty:before {{
        content: attr(placeholder);
        color: #8b949e;
        font-weight: 300;
        cursor: text;
        line-height: 4.6;
    }}
    #single-editor:focus {{
        border-color: #388bfd;
        box-shadow: 0 0 0 2px rgba(56, 139, 253, 0.3);
    }}
    .send-btn {{
        position: absolute;
        right: 30px;
        bottom: 20px;
        width: 32px;
        height: 32px;
        background: #ffffff;
        color: #0e1117;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 900;
        line-height: 1;
        transition: background 0.2s, transform 0.1s;
    }}
    .send-btn:hover {{
        background: #e6edf3;
        transform: scale(1.05);
    }}
    .send-btn:active {{
        transform: scale(0.95);
    }}
</style>
</head>
<body>

<div id="editor-container">
    <div id="single-editor" contenteditable="true" autofocus placeholder="Ask anything..."></div>
    <button class="send-btn" id="send-button" title="Send message">↑</button>
</div>

<script>
function sendStreamlitMessage(type, payload = {{}}) {{
    window.parent.postMessage({{
        isStreamlitMessage: true,
        type: type,
        ...payload
    }}, "*");
}}

if (!window.renderedHandshake) {{
    window.renderedHandshake = true;
    sendStreamlitMessage("streamlit:componentReady", {{ apiVersion: 1 }});
    sendStreamlitMessage("streamlit:setFrameHeight", {{ height: 260 }});
}}

const VOCAB_MAP = {VOCAB_MAP_JSON};
const DEFAULT_B64 = "{DEFAULT_IMAGE_B64}";
const STABILITY_LOOKBACK = {STABILITY_CHAR_WINDOW};
const editor = document.getElementById('single-editor');
const sendBtn = document.getElementById('send-button');

// Master raw text state buffer to avoid contenteditable extraction mutation bugs
let rawInputText = "";
let convertedTokenIndices = new Set();

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
ctx.font = "14px 'IBM Plex Mono', monospace";

function getTextWidth(text) {{
    return Math.max(20, Math.ceil(ctx.measureText(text).width + 12));
}}

function getScaledFontSize(str) {{
    const len = Math.max(1, str.length);
    const calculated = Math.floor(55 / (len * 0.65));
    return Math.min(13, Math.max(9, calculated));
}}

function escapeHtml(str) {{
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/:/g, "&#58;");
}}

function greedyTokenize(text) {{
    if (!text) return [];
    let tokens = [];
    let i = 0;
    
    while (i < text.length) {{
        let longestMatch = "";
        for (let j = Math.min(text.length, i + 20); j > i; j--) {{
            let sub = text.substring(i, j);
            if (VOCAB_MAP[sub]) {{
                longestMatch = sub;
                break;
            }}
        }}
        
        if (longestMatch) {{
            tokens.push(longestMatch);
            i += longestMatch.length;
        }} else {{
            tokens.push(text[i]);
            i += 1;
        }}
    }}
    return tokens;
}}

function cleanLineStartMargins() {{
    const wrappers = editor.querySelectorAll('.word-wrapper');
    let currentTop = -1;

    wrappers.forEach(el => {{
        const top = el.offsetTop;
        if (top > currentTop) {{
            currentTop = top;
            el.style.marginLeft = '0px';
        }}
    }});
}}

function renderDOMFromState() {{
    const fullText = rawInputText;
    if (!fullText) {{
        convertedTokenIndices.clear();
        editor.innerHTML = "";
        return;
    }}

    const L = fullText.length;
    if (L === 0) {{
        convertedTokenIndices.clear();
        editor.innerHTML = "";
        return;
    }}

    // 1. Evaluate prefix tokenization trace across all prefix lengths 1..L
    const prefixTokenizations = [];
    for (let k = 1; k <= L; k++) {{
        const sub = fullText.substring(0, k);
        prefixTokenizations.push(greedyTokenize(sub));
    }}

    // 2. Full string tokenization
    const finalTokens = prefixTokenizations[L - 1];

    // 3. Evaluate stability of each token in the final tokenization
    let stableCount = 0;
    let currentCharPos = 0;

    for (let i = 0; i < finalTokens.length; i++) {{
        const tokText = finalTokens[i];
        const tokEndPos = currentCharPos + tokText.length;

        const minEvalPrefix = tokEndPos;
        const maxEvalPrefix = Math.min(L, tokEndPos + STABILITY_LOOKBACK);

        let isStable = true;

        for (let k = minEvalPrefix; k <= maxEvalPrefix; k++) {{
            const tokensAtK = prefixTokenizations[k - 1];
            if (!tokensAtK || tokensAtK[i] !== tokText) {{
                isStable = false;
                break;
            }}
        }}

        if (L - tokEndPos < STABILITY_LOOKBACK && isStable) {{
            isStable = false;
        }}

        if (isStable && i === stableCount) {{
            stableCount++;
        }} else {{
            break;
        }}

        currentCharPos = tokEndPos;
    }}

    const currentStableIndices = new Set();

    // 4. Construct word-grouped HTML layout from exact raw text state
    let wordsHTML = [];
    let currentWordSpans = [];

    for (let i = 0; i < finalTokens.length; i++) {{
        const part = finalTokens[i];
        if (!part) continue;

        const hasLeadingSpace = part.startsWith(" ") || part.startsWith("Ġ");
        const strippedText = part.trim();
        const escapedRaw = escapeHtml(part);
        const escapedDisplay = escapeHtml(strippedText || part);

        if (hasLeadingSpace && i > 0 && currentWordSpans.length > 0) {{
            const marginLeft = wordsHTML.length > 0 ? 10 : 0;
            wordsHTML.push(`<span class="word-wrapper" style="margin-left: ${{marginLeft}}px;">${{currentWordSpans.join("")}}</span>`);
            currentWordSpans = [];
        }}

        if (i < stableCount) {{
            const tokenData = VOCAB_MAP[part];
            const b64 = tokenData ? tokenData.b64 : DEFAULT_B64;
            const scaledFontSize = getScaledFontSize(strippedText || part);
            const initialTextPx = getTextWidth(part);

            const isNewlyConverted = !convertedTokenIndices.has(i);
            currentStableIndices.add(i);

            const animClass = isNewlyConverted ? "just-converted" : "";

            const spanHTML = `<span class="token-container ${{animClass}}" style="--initial-width: ${{initialTextPx}}px;"><img src="data:image/png;base64,${{b64}}" class="token-img" onerror="this.style.display='none'" /><span class="token-hover-text" data-raw-text="${{escapedRaw}}" style="font-size: ${{scaledFontSize}}px;">${{escapedDisplay}}</span></span>`;
            currentWordSpans.push(spanHTML);
        }} else {{
            // Keep unstable trailing tokens as stripped editable text boxes
            const spanHTML = `<span class="plain-token-text" data-raw-text="${{escapedRaw}}">${{escapedDisplay}}</span>`;
            currentWordSpans.push(spanHTML);
        }}
    }}

    if (currentWordSpans.length > 0) {{
        const marginLeft = wordsHTML.length > 0 ? 10 : 0;
        wordsHTML.push(`<span class="word-wrapper" style="margin-left: ${{marginLeft}}px;">${{currentWordSpans.join("")}}</span>`);
    }}

    // Append trailing active span to receive caret
    wordsHTML.push('<span id="active-text-node"></span>');

    convertedTokenIndices = currentStableIndices;

    editor.innerHTML = wordsHTML.join("");

    cleanLineStartMargins();

    // Move text cursor inside trailing active text node
    const activeNode = document.getElementById('active-text-node');
    if (activeNode) {{
        const range = document.createRange();
        const sel = window.getSelection();
        range.selectNodeContents(activeNode);
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
    }}
}}

// Intercept typing input directly to update raw input text buffer cleanly
editor.addEventListener('input', (e) => {{
    if (e.inputType === "deleteContentBackward") {{
        rawInputText = rawInputText.slice(0, -1);
    }} else if (e.data) {{
        rawInputText += e.data;
    }} else {{
        // Fallback text extraction if input type is unsupported
        let activeNode = document.getElementById('active-text-node');
        if (activeNode && activeNode.innerText) {{
            rawInputText += activeNode.innerText;
        }}
    }}
    renderDOMFromState();
}});

// Handle Backspace and Delete keys reliably
editor.addEventListener('keydown', (e) => {{
    if (e.key === 'Backspace') {{
        e.preventDefault();
        if (rawInputText.length > 0) {{
            rawInputText = rawInputText.slice(0, -1);
            renderDOMFromState();
        }}
    }} else if (e.key === 'Enter') {{
        if (!e.shiftKey) {{
            e.preventDefault();
            submitMessage();
        }}
    }}
}});

function submitMessage() {{
    const val = rawInputText.trim();
    if (val.length > 0) {{
        sendStreamlitMessage("streamlit:setComponentValue", {{ 
            value: JSON.stringify({{ text: val, id: Date.now() }}) 
        }});
        rawInputText = "";
        editor.innerHTML = "";
        convertedTokenIndices.clear();
    }}
}}

sendBtn.addEventListener('click', () => {{
    submitMessage();
}});
</script>
</body>
</html>
"""

index_file = os.path.join(BUILD_DIR, "index.html")
with open(index_file, "w", encoding="utf-8") as f:
    f.write(HTML_CONTENT)

# Top-level global component declaration
visual_token_component = components.declare_component("visual_token_input", path=BUILD_DIR)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_msg_ids" not in st.session_state:
    st.session_state.processed_msg_ids = set()

# 1. Render all conversation history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["html_content"], unsafe_allow_html=True)

# 2. Render Component at bottom container
input_container = st.container()
with input_container:
    user_payload_raw = visual_token_component(key="static_token_widget", default="")

# Unpack message payload safely
clean_prompt = ""
msg_id = None

if user_payload_raw and isinstance(user_payload_raw, str):
    try:
        data = json.loads(user_payload_raw)
        clean_prompt = data.get("text", "").strip()
        msg_id = data.get("id")
    except Exception:
        clean_prompt = user_payload_raw.strip()
        msg_id = clean_prompt

# 3. Process new user input and stream output into the dedicated chat container
if clean_prompt and msg_id and msg_id not in st.session_state.processed_msg_ids:
    st.session_state.processed_msg_ids.add(msg_id)
    
    user_html = render_tokens_html(clean_prompt)
    st.session_state.messages.append({"role": "user", "html_content": user_html, "raw_text": clean_prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_html, unsafe_allow_html=True)

        # Generate Assistant response into chat_container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            chat_history = []
            for msg in st.session_state.messages:
                if "raw_text" in msg:
                    chat_history.append({"role": msg["role"], "content": msg["raw_text"]})
                else:
                    chat_history.append({"role": msg["role"], "content": clean_prompt})
            
            inputs = tokenizer.apply_chat_template(chat_history, return_tensors="pt", add_generation_prompt=True).to(model.device)
            
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids
            elif isinstance(inputs, torch.Tensor):
                input_ids = inputs
            else:
                input_ids = torch.tensor(inputs).to(model.device)
                
            generated_ids = input_ids[0].tolist()
            assistant_tokens = []
            
            with torch.no_grad():
                for _ in range(MAX_GEN_TOKENS):
                    outputs = model(torch.tensor([generated_ids]).to(model.device))
                    next_token_id = int(torch.argmax(outputs.logits[0, -1, :]).item())
                    
                    if next_token_id == tokenizer.eos_token_id:
                        break
                        
                    generated_ids.append(next_token_id)
                    assistant_tokens.append(next_token_id)
                    
                    assistant_text = tokenizer.decode(assistant_tokens)
                    assistant_html = render_tokens_html(assistant_text)
                    
                    # Stream HTML token images cleanly into chat bubble
                    message_placeholder.markdown(assistant_html, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "html_content": assistant_text, "raw_text": assistant_text})