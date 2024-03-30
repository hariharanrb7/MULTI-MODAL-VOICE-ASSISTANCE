#Save the below code with filename 'main.py'
#Run in your terminal
#While running in, enter the command "streamlit run main.py" in your terminal to open the web app in localhost

from dotenv import load_dotenv
load_dotenv() ##load all the environment variables

#libraries used
import streamlit as st
import requests
import os
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import moviepy.editor as mp
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PIL import Image #Python Imaging Library
from youtube_transcript_api import YouTubeTranscriptApi

###

#defining model for text and voice
def get_gemini_repsonse(prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt])
    return response.text

# defining model for image
image=""
def get_gemini_repsonse1(prompt,image):
  model=genai.GenerativeModel('gemini-pro-vision')
  response=model.generate_content([prompt,image[0]])
  return response.text

#check if a file has been uploaded
def input_image_setup(uploaded_file):    
    if uploaded_file is not None: #read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type, #get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    elif uploaded_file is None and prompt or voice_mode is not None:
       return None 
    else:
        raise FileNotFoundError("No file uploaded")

#define the function to convert text to speech using pyttsx3
def text_to_speech(text):
  engine = pyttsx3.init() #set the voice to female
  eng = pyttsx3.init() #initialize an instance
  voice = eng.getProperty('voices')
  engine.setProperty('voice', voice[1].id)
  engine.say(text)
  engine.runAndWait()


###
input_prompt = """You are an expert in everything and an 
excellent Personal Assistant where you also need to 
analyse the image or question asked by me  
and response in more relevantly,formally,in short manner within 50 words or less, Maintaining context while responding, 
don't use symbols like bullet points, asterics etc., like below format """


## webapp using streamlit  
st.title("Multimodal Voice Assistant App") #set the title and the sidebar of the web app
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
voice_mode = st.checkbox("Enable voice mode",value = True) #create a checkbox to enable or disable voice input and output
prompt = st.text_input("Type or say something") #create a text input for the user to type their query or command
uploaded_file = st.file_uploader("Choose an image or a video...", type=["jpg", "jpeg", "png"])#create a image input for the user to upload images

image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

#check if voice mode is enabled
if voice_mode: #create a button to start listening to the user's voice
    if st.button("Start Listening"): #initialize the speech recognizer
      recognizer = sr.Recognizer() #use the default microphone as the audio source
      with sr.Microphone() as source: #adjust the ambient noise
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening...")       
        audio = recognizer.listen(source) #listen to the user's voice
        st.info("Processing...") #display a message to indicate processing
#recognize the user's voice using google speech recognition
        try:
           prompt = recognizer.recognize_google(audio)
           st.write(f"You: {prompt}") #display the prompt as the user message
           response=get_gemini_repsonse(input_prompt+prompt)
           st.write(f"Voice Assistant : {response}")#display the response as the voice assistant's message
           text_to_speech(response) 
        except:
           st.error("Sorry, I could not understand your voice. Please try again.") #display an error message

submit=st.button("Submit") #create a button to submit the query or command

if prompt: #check if the prompt is not empty
   st.write(f"You: {prompt}") #display the prompt as the user's message
 


if submit:
  if uploaded_file is None and (prompt or voice_mode) is not None:  #if text or voice is submitted
       response= get_gemini_repsonse(input_prompt+prompt)
       st.write(f"Voice Assistant: {response}")#display the response as the voice assistant's message
       text_to_speech(response)
  elif uploaded_file is not None and (prompt or voice_mode) is None: #if image only is submitted
       image=input_image_setup(uploaded_file)
       response= get_gemini_repsonse1(input_prompt+prompt,image)
       st.write(f"Voice Assistant: {response}")#display the response as the voice assistant's message
       text_to_speech(response)
  elif uploaded_file is not None and (prompt or voice_mode) is not None: #if both are submitted
       image=input_image_setup(uploaded_file)
       response=get_gemini_repsonse1(input_prompt+prompt,image)
       st.write(f"Voice Assistant: {response}")#display the response as the voice assistant's message
       text_to_speech(response)  
 

elif uploaded_file is not None and (prompt or voice_mode) is not None: # if nothing is not submitted
  st.warning("Please type, say or upload anything")

### PDF READER
with st.sidebar:
    PDF_reader = st.checkbox("Enable PDF Reader",value = True)
if PDF_reader:
    #defining for Functions for PDF
    def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader= PdfReader(pdf)
            for page in pdf_reader.pages:
                text+= page.extract_text()
        return  text


 #defining function that break down large text documents into pieces
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
#defining function to process text chunks, generates embeddings, and creates a vector store for efficient similarity search.
    def get_vector_store(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

# defining model for PDF reader
    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro",
                                temperature=0.3)

        prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        return chain

#defining function that search user question within the pdf uploaded and response
    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])
        text_to_speech(response["output_text"])
    ###
     #defining webapp for PDF reader
    def main():
        st.header("Multiple PDF Reader")
        st.markdown("##### Upload Multiple PDFs and Ask Questions")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True,help="Please uplaod the pdf")
        if st.button("Submit & Process"):
         with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question) #text input to ask question


    if __name__ == "__main__":
        main()

### Video Analyser 
with st.sidebar:
    video_summarizer = st.checkbox("Enable Video Summarizer",value = True) #checkbox to enable video transcriber
if video_summarizer:
    #defining function to process uploaded video and extract audio
    def process_video(uploaded_video):
        if uploaded_video is not None:
            video_bytes = uploaded_video.read()
            with open("temp_video.mp4", "wb") as f:
                f.write(video_bytes)
            video_clip = mp.VideoFileClip("temp_video.mp4")
            audio_clip = video_clip.audio #extract audio from video
            audio_clip.write_audiofile("temp_audio.wav")
            return "temp_audio.wav"
        else:
            raise FileNotFoundError("No video uploaded")

    #defining function to transcribe audio into text
    def transcribe_audio(audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    
    input_prompt1= """You are video summarizer. You will be taking the video
        and summarizing the entire video and providing the important summary in points
        within 250 words. Please provide the summary of the text given here:  """
    
    # webapp code for video analysis and response generation
    st.header("Video Analyzer")
    st.markdown("##### Upload a Video File (that contains audio) to Analyze")

    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi"]) # creating video input("mp4" or "avi")

    if uploaded_video is not None: # if video uploaded
        audio_file = process_video(uploaded_video) 
        st.success("Video uploaded and processed successfully!")
        speak_mode = st.button("Through speak mode") 
        if st.button("Analyze and get summary"): #creating button to analysis and if it is pressed
            audio_text = transcribe_audio(audio_file)    
            response = get_gemini_repsonse(input_prompt1 + audio_text) #getting response
            st.markdown("## Video Analysis Summary:")
            st.write(response)
        if speak_mode:
            audio_text = transcribe_audio(audio_file)    
            response = get_gemini_repsonse(input_prompt1 + audio_text) #getting response
            st.markdown("## Video Analysis Summary:")
            st.write(response)
            text_to_speech(response) #response in voice mode
    
### YT Transcriber
with st.sidebar:
    yt_transcript = st.checkbox("Enable YT Transcriber",value = True) #creating check box for YT transcriber
if yt_transcript:  # if YT transcriber is enabled
    prompt_for_YT_Videos="""You are Yotube video summarizer. You will be taking the transcript text
    and summarizing the entire video and providing the important summary in points 
    within 250 words without using symbols like asterics, bullet points etc., Please provide the summary of the text given here:  """             #prompt


    ## getting the transcript data from yt videos
    def extract_transcript_details(youtube_video_url):
        try:
            video_id=youtube_video_url.split("=")[1]
            
            transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

            transcript = ""
            for i in transcript_text:
                transcript += " " + i["text"]

            return transcript

        except Exception as e:
            raise e
        
    #defining model for YT Transcriber
    def generate_gemini_content(transcript_text,prompt_for_YT_Videos):

        model=genai.GenerativeModel("gemini-pro")
        response=model.generate_content(prompt_for_YT_Videos+transcript_text)  ## getting the summary based on Prompt from Google Gemini Pro
        return response.text

    st.header("YouTube Transcriber")
    st.markdown("##### Upload Youtube link to get description")
    youtube_link = st.text_input("Enter YouTube Video Link:") # creating input for uploading youtube link

    if youtube_link:         #processing the youtube link
        video_id = youtube_link.split("=")[1]
        print(video_id)
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True) #showing the thumbnail of the video link provided
    speak_mode1 = st.button("Get summary in speak mode") 
    if st.button("Get Summary"):
        transcript_text=extract_transcript_details(youtube_link)  #getting response

        if transcript_text:
            summary=generate_gemini_content(transcript_text,prompt_for_YT_Videos)#getting response
            st.markdown("## Detailed Notes:")
            st.write(summary)       
    if speak_mode1:
        transcript_text=extract_transcript_details(youtube_link)  #getting response
        if transcript_text:
            summary=generate_gemini_content(transcript_text,prompt_for_YT_Videos)#getting response
            st.markdown("## Detailed Notes:")
            st.write(summary)
            text_to_speech(summary) #response in voice mode
                
            

### HEALTH ADVISOR
with st.sidebar:
    health_advisor = st.checkbox("Enable Health Advisor",value = True)
if health_advisor:
    def get_gemini_repsonse(prompt,image):
        model=genai.GenerativeModel('gemini-pro-vision')
        response=model.generate_content([prompt,image[0]])
        return response.text

    def input_image_setup(uploaded_file):
        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.getvalue()

            image_parts = [
                {
                    "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")

    ##initialize our streamlit app

    st.header("Health Advisor")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


    submit=st.button("Tell me the total calories")

    input_prompt="""
    You are an expert in nutritionist where you need to see the food items from the image
                and calculate the total calories, also provide the details of every food items with calories intake
                is below format

                1. Item 1 - no of calories
                2. Item 2 - no of calories
                ----
                ----
            Finally you can also mention whether the food is healthy or not and also mention the percentage split of 
            the ratio of carbohydrates, fats, fiber, sugar, and other important things required in our diet.

            Also if the uploaded image is not the food items please mention to upload only food items image to track calories


    """

    ## If submit button is clicked

    if submit:
        image_data=input_image_setup(uploaded_file)
        response=get_gemini_repsonse(input_prompt,image_data)
        st.subheader("Your Food Contains")
        st.write(response)
        text_to_speech(response)
