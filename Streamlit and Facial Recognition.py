import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import os

# --------------------------------
# DATASET PATH
# --------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "dataset")

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="AI Facial Recognition System",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.main-title {
    text-align: center;
    color: white;
    font-size: 50px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------

st.markdown(
    "<div class='main-title'>🧠 AI Facial Recognition System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Internship Project - Facial Recognition Using Deep Learning</div>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.header("📊 Dataset Information")

    if os.path.exists(dataset_path):

        dataset_size = len([
            file
            for file in os.listdir(dataset_path)
            if file.endswith((".jpg", ".jpeg", ".png"))
        ])

        st.metric(
            "Total Dataset Images",
            dataset_size
        )

        st.success("System Online")

    else:

        st.error("Dataset folder not found")

# --------------------------------
# FILE UPLOADER
# --------------------------------

uploaded_file = st.file_uploader(
    "📸 Upload Test Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# FACE RECOGNITION
# --------------------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp:

        image.save(temp.name)

        temp_path = temp.name

    with st.spinner("🔍 Searching Dataset..."):

        try:

            result = DeepFace.find(
                img_path=temp_path,
                db_path=dataset_path,
                enforce_detection=False,
                silent=True
            )

            if len(result[0]) > 0:

                best_match = result[0].iloc[0]

                matched_path = best_match["identity"]

                filename = os.path.basename(
                    matched_path
                )

                person_name = os.path.splitext(
                    filename
                )[0]

                distance = best_match["distance"]

                confidence = max(
                    0,
                    (1 - distance) * 100
                )

                with col2:

                    st.subheader("Recognition Result")

                    st.success(
                        "✅ Person Identified"
                    )

                    st.markdown(
                        f"### 👤 {person_name}"
                    )

                    st.write(
                        f"Confidence Score: {confidence:.2f}%"
                    )

                    st.progress(
                        min(
                            int(confidence),
                            100
                        )
                    )

                    st.image(
                        matched_path,
                        caption="Best Match From Dataset",
                        use_container_width=True
                    )

            else:

                with col2:

                    st.error(
                        "❌ No Match Found"
                    )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Internship Project | Streamlit + DeepFace Facial Recognition"
)