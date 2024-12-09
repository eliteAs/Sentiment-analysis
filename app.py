import streamlit as st
import sentiment_mod as s

# Streamlit app configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Sentiment Analysis App")
st.markdown("Analyze the sentiment of your text input and get confidence levels for predictions.")

# User input section
st.sidebar.header("Settings")
sentiment_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.01)
st.sidebar.markdown("Adjust the confidence threshold to filter predictions.")

st.write("### Enter Your Text")
user_input = st.text_area("Type something to analyze its sentiment:")

# Analysis result section
if st.button("Analyze"):
    if user_input.strip():
        label, confidence = s.sentiment(user_input)
        confidence_percentage = confidence * 100

        # Display results
        if confidence >= sentiment_threshold:
            st.success(f"**Predicted Sentiment:** {label}")
            st.info(f"**Confidence Level:** {confidence_percentage:.2f}%")
        else:
            st.warning(
                f"Prediction confidence ({confidence_percentage:.2f}%) is below the threshold ({sentiment_threshold * 100:.2f}%)."
            )
    else:
        st.error("Please enter some text for analysis!")

# Footer
st.markdown(
    """
    ---
    Under development phase!!
    """
)
