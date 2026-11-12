import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from scipy.ndimage import shift as nd_shift
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    from streamlit_drawable_canvas import st_canvas  # pip install streamlit-drawable-canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

st.set_page_config(page_title="Digit Lab", page_icon="✍️", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #1d1b3a 0%, #0e1020 55%, #0a0b16 100%);
    }
    h1, h2, h3, p, label, span, div { color: #e8e9f5; }
    .hero h1 {
        font-size: 2.8rem; font-weight: 800; margin-bottom: 0;
        background: linear-gradient(90deg, #7c83ff, #38e1c6 60%, #ffd166);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #9aa0c7; font-size: 1.05rem; margin-top: .2rem; }
    .panel {
        background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px; padding: 1.2rem 1.4rem; backdrop-filter: blur(6px);
    }
    .digit-card { text-align: center; padding: 1.4rem 1rem; }
    .digit {
        font-size: 7rem; line-height: 1; font-weight: 800;
        background: linear-gradient(135deg, #7c83ff, #38e1c6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .conf { color: #9aa0c7; font-size: 1.1rem; }
    .badge {
        display: inline-block; padding: .2rem .7rem; border-radius: 999px;
        font-size: .8rem; font-weight: 600; margin-top: .5rem;
    }
    .good { background: rgba(56,225,198,.15); color: #38e1c6; }
    .mid  { background: rgba(255,209,102,.15); color: #ffd166; }
    .low  { background: rgba(255,107,129,.15); color: #ff6b81; }
    [data-testid="stSidebar"] { background: #0b0c1a; border-right: 1px solid rgba(255,255,255,.06); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model ----------
def _augment(X, y, rng):
    """Add small shifts so the model tolerates off-center digits."""
    imgs = X.reshape(-1, 8, 8)
    out_x, out_y = [imgs], [y]
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        out_x.append(np.array([nd_shift(i, (dy, dx), cval=0, order=0) for i in imgs]))
        out_y.append(y)
    X_aug = np.concatenate(out_x).reshape(-1, 64)
    y_aug = np.concatenate(out_y)
    idx = rng.permutation(len(y_aug))
    return X_aug[idx], y_aug[idx]


@st.cache_resource(show_spinner="Training the model…")
def load_model():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = _augment(X_train, y_train, np.random.default_rng(42))
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=400, alpha=1e-3, random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


model, test_acc = load_model()


# ---------- Image preprocessing ----------
def preprocess(image: Image.Image):
    """Turn any image into the 8x8, white-on-black format the model was trained on.

    Returns (model_input of shape (1, 64), 8x8 array in 0..1) or (None, None) if blank.
    """
    if image.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image = bg
    gray = ImageOps.autocontrast(image.convert("L"))
    arr = np.array(gray, dtype=np.float32)

    # Digits in the training set are bright on dark. Invert if the border is bright.
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:
        arr = 255 - arr

    arr[arr < 40] = 0  # kill faint noise
    if arr.max() == 0:
        return None, None

    # Crop to the digit, pad to a square with a margin, then downsample.
    ys, xs = np.where(arr > 0)
    arr = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = arr.shape
    side = int(max(h, w) * 1.35)
    canvas = np.zeros((side, side), dtype=np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy: oy + h, ox: ox + w] = arr

    small = Image.fromarray(canvas.astype(np.uint8)).resize((8, 8), Image.LANCZOS)
    small = np.array(small, dtype=np.float32)
    small = small / small.max()
    return small.reshape(1, 64), small


def predict(image: Image.Image):
    x, small = preprocess(image)
    if x is None:
        st.info("That image looks blank. Try a darker or larger digit.")
        return
    probs = model.predict_proba(x)[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        badge = ("good", "High confidence") if conf > 0.85 else (
            ("mid", "Fairly sure") if conf > 0.55 else ("low", "Not sure")
        )
        st.markdown(
            f"""
            <div class="panel digit-card">
              <div class="conf">I think this is a</div>
              <div class="digit">{pred}</div>
              <div class="conf">{conf:.1%} confident</div>
              <span class="badge {badge[0]}">{badge[1]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("What the model sees (8×8 pixels)")
        preview = Image.fromarray((small * 255).astype(np.uint8)).resize(
            (160, 160), Image.NEAREST
        )
        st.image(preview)
    with right:
        st.markdown("##### Probability by digit")
        df = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability": probs})
        st.bar_chart(df, x="Digit", y="Probability", color="#7c83ff", height=300)
        top3 = np.argsort(probs)[::-1][:3]
        st.markdown(
            "**Top guesses:** "
            + "  ·  ".join(f"`{d}` {probs[d]:.1%}" for d in top3)
        )


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>✍️ Digit Lab</h1>
      <p>Draw or upload a handwritten digit and watch a neural network read it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.metric("Model accuracy (held-out test set)", f"{test_acc:.1%}")
    st.caption("Neural network (128-64 hidden units) trained on scikit-learn's 8×8 digits dataset with shift augmentation.")
    st.header("Tips for best results")
    st.markdown(
        """
- Draw one digit, big and bold
- Use a plain background
- Thick strokes work better than thin ones
- Press **Clear** in the canvas toolbar to retry
        """
    )
    stroke = st.slider("Brush size", 10, 40, 22) if HAS_CANVAS else 22

# ---------- Input ----------
tab_draw, tab_upload = st.tabs(["🖌️ Draw", "📁 Upload"])

with tab_draw:
    if not HAS_CANVAS:
        st.warning(
            "Drawing needs an extra package. Install it with "
            "`pip install streamlit-drawable-canvas` and restart the app."
        )
    else:
        c1, c2 = st.columns([1, 1.6], gap="large")
        with c1:
            result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=stroke,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        with c2:
            if result.image_data is not None and result.image_data[:, :, :3].max() > 0:
                predict(Image.fromarray(result.image_data.astype("uint8"), "RGBA"))
            else:
                st.markdown(
                    '<div class="panel">Draw a digit from 0 to 9 on the canvas. '
                    "The prediction appears here as you draw.</div>",
                    unsafe_allow_html=True,
                )

with tab_upload:
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            u1, u2 = st.columns([1, 2.2], gap="large")
            with u1:
                st.image(img, caption="Your image", use_container_width=True)
            with u2:
                predict(img)
        except Exception as e:
            st.error(f"Couldn't read that image: {e}")
    else:
        st.markdown(
            '<div class="panel">Upload a PNG or JPG with a single digit. '
            "Light-on-dark and dark-on-light both work.</div>",
            unsafe_allow_html=True,
        )
