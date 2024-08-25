import streamlit as st
#from backend.solution import answer_user_question
from PIL import Image

def answer_user_question(dummy, dymmy):
    
    return "This is a dummy answer"

# Function to load and display the logo
def display_logo(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, use_column_width=True)

# Set page configuration
st.set_page_config(page_title="DGCL Gen-AI", layout="centered")

# Charger l'image
image = "logo_white.png"
width = 180

st.markdown(
    f"""
    <style>
        img {{
	display: flex; 
	justify-content: center;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Center the image using Streamlit's layout features
col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
with col3:
    st.image(image, width=width)




st.markdown(
    """
    <style>
       .title-text {
        color: #FFFFFF;
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        font-family: Cairo;  
	background-color: rgba(218,172,151, 0.0);
    }
    
    .subtitle-text {
        color: #FFFFFF; 
        text-align: center;
        font-size: 28px;
	font-weight: bold;
        font-family: Cairo; 
	background-color: rgba(218,172,151, 0.0);
    }
    body, p, label, input, select {
        font-family: Cairo; 
	font-size: 18px;
	font-weight: bold;
        color: #55142D; 
    }



    # .stApp {
    #     background-color: #C17250; 
    #     color: #260f02; 
    #     font-family: Cairo;
    # }

    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.0); 
        color: #260f02;
    }

    .stButton button {
        background-color: rgba(244, 230, 222, 0.0);
	color: white;
    }

   .stBlock {
    background-color: #AED6F1;
}
    h1, h2 {
        text-align: center;
        color: #55142d;
    }
    .st-bb {
        background-color: #CD8C6F;
    }
    .st-f7 {
        background-color: #F4E6DE;
    }
    div[data-testid="stForm"] {
	background-color: rgba(218,172,151, 0.85);
	#244, 230, 222
	
 }
    
    </style>
    """,
    unsafe_allow_html=True,
)


import base64


st.markdown(
    """
    <style>
    .stApp {
        background-image: url(data:image/png;base64,{});
        background-size: cover;
        color: #260f02; 
        font-family: Cairo;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the background image
with open("BG_dark_40.png", "rb") as file:
    btn = file.read()
b64 = base64.b64encode(btn).decode()


st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{b64});
        background-size: cover;
        color: #260f02; 
        font-family: Cairo;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<p class="title-text">Historical Generative AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Only One Diriyah</p>', unsafe_allow_html=True)
st.divider()

# Form and user interaction
with st.form("questions"):
    greetings_ph = st.empty()
    lang = st.radio(
        "Please select the Model language:",
        ["عربي", "English"],
        captions=["", ""],
        horizontal=True,
    )
    question = st.text_input("Please write your question here: ", max_chars=300)
    cols = st.columns(4, gap="large")
    with cols[-1]:
        submitted = cols[-1].form_submit_button("Submit")

    if submitted:

        with st.spinner('I am working on it...'):
            tgt_lang = 'ar' if lang == 'عربي' else 'en'
            result = answer_user_question(question, tgt_lang)
        st.write(f"**Answer:**\n{result['response']}")
        
        with  st.expander("Show the response details!"):
            
            src_docs = ' '.join( x.page_content  for x in  result['source_documents'])

            st.write(f"**This text was used to find an answer:**\n\n{src_docs}")

            #st.write(f"\n**Original Response:**\n\n{result['result']}")
