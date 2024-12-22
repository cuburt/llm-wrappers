import streamlit as st
from PIL import Image
from chain import Chain
import base64


@st.cache_resource
def load_backend():
    chain = Chain()
    return chain


def main():
    chain = load_backend()
    st.title("UI Image Digitization and Recommendation")

    # Sidebar for input method selection
    input_method = st.sidebar.radio("Choose input method:", ("Upload Image", "Enter Text"))
    model = st.sidebar.radio("Choose Model:", ("Gemini 1.5 Pro", "GPT4o"))
    model = "gpt" if model == "GPT4o" else "gemini"
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:

            image = f"data:{uploaded_file.type};base64,{base64.b64encode(uploaded_file.getvalue()).decode('ascii')}"
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    response, similar_images = chain(image=image, model=model)
                display_conversation(response, similar_images)


    else:  # Enter Text
        text_input = st.text_input("Enter your query:")
        if text_input and st.button("Submit Query"):
            with st.spinner("Processing..."):
                response, similar_images = chain(question=text_input)
            display_conversation(response, similar_images)


def display_conversation(response, similar_images=None):
    st.subheader("Result:")
    st.markdown(response)
    if similar_images:
        st.subheader("Relevant UI Designs")
        for i, col in enumerate(st.columns(len(similar_images))):
            vars()[f"col_{i}"] = col
        for i, t in enumerate(similar_images):
            col = vars()[f"col_{str(i)}"]
            col.image(t, use_column_width="always")



if __name__ == "__main__":
    main()
