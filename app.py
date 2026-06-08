import streamlit as st 
from translation import translation
import contractions




st.set_page_config(
    page_title="English to Hindi Translator",
    page_icon="🌐",
    layout="centered"
)
st.title("🌐 English to Hindi Neural Translator")
st.write("Type a sentence in English below to get its Hindi translation using your trained deep learning model.")
st.markdown("---")


user_input=st.text_input("Enter a English text ",placeholder='how are you?')

if st.button('Transalte'):
    if user_input=="":
        st.warning('Please enter the text to translate')
    else:
        with st.spinner('Analyzing text structure'):
            if user_input:
                try:
                    output=translation(user_input)
                    st.subheader('Hindi Translation')
                    st.success(output)

                except Exception as e:
                    st.error(f"An error occurred during inference: {e}")    

                