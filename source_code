!pip install langchain
# !pip install langchain-community
# !pip install langchain-openai
!pip install -U langchain-community
!pip install -q --upgrade google-generativeai langchain-google-genai
!pip install google-auth
!pip install chromadb
!pip install git+https://github.com/openai/whisper.git

#from google.colab import drive
#drive.mount('/content/drive')

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_google_genai.embeddings import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.evaluation.loading import load_evaluator
# import openai
# from dotenv import load_dotenv
import os
import shutil


DATA_PATH="enter your data path here"

from google.oauth2 import service_account
credentials_path = 'enter your credentials part here'
credentials = service_account.Credentials.from_service_account_file(credentials_path)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",credentials=credentials)
# Get embedding for a word.
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001",credentials=credentials)
vector = embedding_function.embed_query("apple")
print(f"Vector for 'apple': {vector}")
print(f"Vector length: {len(vector)}")

    # Compare vector of two words
word1, word2 = "apple", "iphone"

# Get their embeddings
vector1 = embedding_function.embed_query(word1)
vector2 = embedding_function.embed_query(word2)

# Convert to numpy arrays
vector1 = np.array(vector1).reshape(1, -1)  # Reshape to 2D for sklearn
vector2 = np.array(vector2).reshape(1, -1)

# Compute Cosine Similarity
similarity = cosine_similarity(vector1, vector2)[0][0]

print(f"Cosine Similarity between '{word1}' and '{word2}': {similarity:.4f}")

CHROMA_PATH = "chroma"
def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


    main()
#Taken from Deepseek
# Install required library
!pip install pydub -q

# Import dependencies
from IPython.display import HTML, Javascript, Audio
from google.colab import output
from base64 import b64decode
from pydub import AudioSegment

# Define JavaScript/HTML interface
js_code = """
<div id="recorder">
  <button onclick="startRecording()" id="startBtn">🎤 Start Recording</button>
  <button onclick="stopRecording()" id="stopBtn" disabled>⏹ Stop & Save</button>
</div>

<script>
let mediaRecorder;
let audioChunks = [];

async function startRecording() {
  document.getElementById("stopBtn").disabled = false;
  document.getElementById("startBtn").disabled = true;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.start();
  } catch (error) {
    alert('Error accessing microphone: ' + error.message);
  }
}

async function stopRecording() {
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;

  mediaRecorder.stop();
  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const reader = new FileReader();

    reader.readAsDataURL(audioBlob);
    reader.onloadend = () => {
      const base64data = reader.result;
      google.colab.kernel.invokeFunction('notebook.saveAudio', [base64data], {});
    };

    audioChunks = [];
  };
}
</script>
"""

# Audio processing callback
def save_audio(base64_data):
    try:
        # Decode base64 audio data
        audio_bytes = b64decode(base64_data.split(',')[1])

        # Save original webm file
        with open('my_recording.webm', 'wb') as f:
            f.write(audio_bytes)

        # Convert to WAV format
        audio = AudioSegment.from_file('my_recording.webm', format='webm')
        audio.export('my_recording.wav', format='wav')

        # Show audio player
        display(Audio('my_recording.wav'))
        print("✅ Audio saved as my_recording.wav")

    except Exception as e:
        print("Error processing audio:", e)

# Register callback
output.register_callback('notebook.saveAudio', save_audio)

# Display the recorder interface
display(HTML(js_code))

# # Optional: Add download button
# def add_download():
#     display(HTML(
#         '''<div style="margin-top:20px">
#            <a href="my_recording.wav" download>
#              <button>⬇ Download WAV File</button>
#            </a>
#            </div>'''
#     ))

# # Call this after saving audio to show download button
# add_download()

import whisper
query=whisper.load_model("medium").transcribe("my_recording.wav",language="en")['text']
print(query)

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001",credentials=credentials)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
from langchain.chains import RetrievalQA
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
response = qa_chain.invoke({"query": query})
print("\nGenerated Response:\n", response["result"])
