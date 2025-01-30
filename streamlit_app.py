import pandas as pd
import base64
import streamlit as st
import json
import time
from io import StringIO, 
import io
import requests  # pip install requests
from streamlit_lottie import st_lottie  
import vertexai
from typing import List
from google.cloud import translate, aiplatform
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import os

generation_config = {
    "candidate_count": 1,
    "max_output_tokens": 8192,
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 1,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]
def generate(text, src, trg, llm_model, tone='formal', domain='Healthcare', instruction='0'):
    # Initialize Vertex AI with project and location from secrets
    service_account_info = st.secrets["gcp_service_account"]
    vertexai.init(
        project = service_account_info["project_id"],
        location = "us-central1",
        credentials = service_account_key.json,
    )
    
    model = GenerativeModel(
        model_name=llm_model
    )

    # Generate content
    responses = model.generate_content(
        [f'You are an expert Translator. You are tasked to translate documents from {src} to {trg}. \
         Please provide an accurate translation of this text which is from {domain} and return translation text only, considering the {tone} \
         Instruction: {instruction} \
         :{text}'],
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    return responses.candidates[0].content.parts[0].text


def get_transcript(audio_file, audio_language = 'unknown'):
    url = "https://api.sarvam.ai/speech-to-text"

    files = {
        "file": (audio_file.name, audio_file, audio_file.type)  # Directly use UploadedFile object
    }
    

    data = {
        "language_code": audio_language,
        "model": "saarika:v2",
        "with_diarization": "true",
        "with_timestamps": "true"
    }

    headers = {
        "api-subscription-key": "5a73b765-cbce-43bd-8080-c7430ce4d961"
    }

    response = requests.post(url, files=files, data=data, headers=headers)

    return response

def generate_NMT(strs_to_translate: List[str], src: str, tgt: str
) -> translate.TranslationServiceClient:
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "us-central1"

    parent = f"projects/lisanai/locations/{location}"

    # Translate text from en to fr
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": strs_to_translate,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": src,
            "target_language_code":  tgt,
        }
    )

    return [text.translated_text for text in response.translations]  

def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
c1, c2, c3 = st.columns([2,5,1], vertical_alignment="center")
lottie_hello = load_lottieurl("https://lottie.host/057e0efe-27c7-4397-840c-f1f25b8a682a/6Dw9TLkyW5.json")
with c2:
    st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        #renderer="svg", # canvas
        height=300,
        width=300,
        key=None,

    )
a1, a2, a3 = st.columns([1,3,1], vertical_alignment="center")
with a2:
    #st.title("Your AI for your Documents")
    st.markdown("<h1 style='text-align: center;'>Lisan AI</h1>", unsafe_allow_html=True)

audio_on = st.toggle("Audio")

if audio_on :
    audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "wave", "x-wav", "mpeg"])

    audio_language_dict = {
        "Unknown": "unknown",
        "Hindi": "hi-IN",
        "Bengali": "bn-IN",
        "Kannada": "kn-IN",
        "Malayalam": "ml-IN",
        "Marathi": "mr-IN",
        "Odia": "od-IN",
        "Punjabi": "pa-IN",
        "Tamil": "ta-IN",
        "Telugu": "te-IN",
        "English (India)": "en-IN",
        "Gujarati": "gu-IN"
    }

    language_opts = tuple(audio_language_dict.keys())
    audio_language =st.selectbox(
            "Select the input Audio Language",
            language_opts,
            index=0
        )
    
    if st.button("Get Transcript"):
        start_time, end_time, speaker_id, transcript =[], [], [], []
        response = get_transcript(audio, audio_language_dict[audio_language])
        for i in eval(response.text)['diarized_transcript']['entries']:
              start_time.append(i['start_time_seconds'])
              end_time.append(i['end_time_seconds'])
              speaker_id.append(i['speaker_id'])
              transcript.append(i['transcript'])
        data = {
                "Start Time": start_time,
                "End Time": end_time,
                "Speaker IDs": speaker_id,
                "Transcripts": transcript
            }
        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
            writer.close()
        
        # Create download button
        st.download_button(
            label="Download Transcript",
            data=output.getvalue(),
            file_name="Transcript.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
else:
    b1, b2 = st.columns([1,1], vertical_alignment="center")
    languages = {
        "Abkhaz": "ab",
        "Acehnese": "ace",
        "Acholi": "ach",
        "Afrikaans": "af",
        "Albanian": "sq",
        "Alur": "alz",
        "Amharic": "am",
        "Arabic": "ar",
        "Armenian": "hy",
        "Assamese": "as",
        "Awadhi": "awa",
        "Aymara": "ay",
        "Azerbaijani": "az",
        "Balinese": "ban",
        "Bambara": "bm",
        "Bashkir": "ba",
        "Basque": "eu",
        "Batak Karo": "btx",
        "Batak Simalungun": "bts",
        "Batak Toba": "bbc",
        "Belarusian": "be",
        "Bemba": "bem",
        "Bengali": "bn",
        "Betawi": "bew",
        "Bhojpuri": "bho",
        "Bikol": "bik",
        "Bosnian": "bs",
        "Breton": "br",
        "Bulgarian": "bg",
        "Buryat": "bua",
        "Cantonese": "yue",
        "Catalan": "ca",
        "Cebuano": "ceb",
        "Chichewa (Nyanja)": "ny",
        "Chinese (Simplified)": "zh-CN or zh (BCP-47)",
        "Chinese (Traditional)": "zh-TW (BCP-47)",
        "Chuvash": "cv",
        "Corsican": "co",
        "Crimean Tatar": "crh",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dinka": "din",
        "Divehi": "dv",
        "Dogri": "doi",
        "Dombe": "dov",
        "Dutch": "nl",
        "Dzongkha": "dz",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Ewe": "ee",
        "Fijian": "fj",
        "Filipino (Tagalog)": "fil or tl",
        "Finnish": "fi",
        "French": "fr",
        "French (French)": "fr-FR",
        "French (Canadian)": "fr-CA",
        "Frisian": "fy",
        "Fulfulde": "ff",
        "Ga": "gaa",
        "Galician": "gl",
        "Ganda (Luganda)": "lg",
        "Georgian": "ka",
        "German": "de",
        "Greek": "el",
        "Guarani": "gn",
        "Gujarati": "gu",
        "Haitian Creole": "ht",
        "Hakha Chin": "cnh",
        "Hausa": "ha",
        "Hawaiian": "haw",
        "Hebrew": "iw or he",
        "Hiligaynon": "hil",
        "Hindi": "hi",
        "Hmong": "hmn",
        "Hungarian": "hu",
        "Hunsrik": "hrx",
        "Icelandic": "is",
        "Igbo": "ig",
        "Iloko": "ilo",
        "Indonesian": "id",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jw or jv",
        "Kannada": "kn",
        "Kapampangan": "pam",
        "Kazakh": "kk",
        "Khmer": "km",
        "Kiga": "cgg",
        "Kinyarwanda": "rw",
        "Kituba": "ktu",
        "Konkani": "gom",
        "Korean": "ko",
        "Krio": "kri",
        "Kurdish (Kurmanji)": "ku",
        "Kurdish (Sorani)": "ckb",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latgalian": "ltg",
        "Latin": "la",
        "Latvian": "lv",
        "Ligurian": "lij",
        "Limburgan": "li",
        "Lingala": "ln",
        "Lithuanian": "lt",
        "Lombard": "lmo",
        "Luo": "luo",
        "Luxembourgish": "lb",
        "Macedonian": "mk",
        "Maithili": "mai",
        "Makassar": "mak",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malay (Jawi)": "ms-Arab",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Maori": "mi",
        "Marathi": "mr",
        "Meadow Mari": "chm",
        "Meiteilon (Manipuri)": "mni-Mtei",
        "Minang": "min",
        "Mizo": "lus",
        "Mongolian": "mn",
        "Myanmar (Burmese)": "my",
        "Ndebele (South)": "nr",
        "Nepalbhasa (Newari)": "new",
        "Nepali": "ne",
        "Northern Sotho (Sepedi)": "nso",
        "Norwegian": "no",
        "Nuer": "nus",
        "Occitan": "oc",
        "Odia (Oriya)": "or",
        "Oromo": "om",
        "Pangasinan": "pag",
        "Papiamento": "pap",
        "Pashto": "ps",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Portuguese (Portugal)": "pt-PT",
        "Portuguese (Brazil)": "pt-BR",
        "Punjabi": "pa",
        "Punjabi (Shahmukhi)": "pa-Arab",
        "Quechua": "qu",
        "Romani": "rom",
        "Romanian": "ro",
        "Rundi": "rn",
        "Russian": "ru",
        "Samoan": "sm",
        "Sango": "sg",
        "Sanskrit": "sa",
        "Scots Gaelic": "gd",
        "Serbian": "sr",
        "Sesotho": "st",
        "Seychellois Creole": "crs",
        "Shan": "shn",
        "Shona": "sn",
        "Sicilian": "scn",
        "Silesian": "szl",
        "Sindhi": "sd",
        "Sinhala (Sinhalese)": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swati": "ss",
        "Swedish": "sv",
        "Tajik": "tg",
        "Tamil": "ta",
        "Tatar": "tt",
        "Telugu": "te",
        "Tetum": "tet",
        "Thai": "th",
        "Tigrinya": "ti",
        "Tsonga": "ts",
        "Tswana": "tn",
        "Turkish": "tr",
        "Turkmen": "tk",
        "Twi (Akan)": "ak",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Xhosa": "xh",
        "Yiddish": "yi",
        "Yoruba": "yo",
        "Yucatec Maya": "yua",
        "Zulu": "zu"
    }
    keys_lang = tuple(languages.keys())


    llm_model =st.selectbox(
            "Model Selection",
            ("gemini-1.5-flash-002", "gemini-1.5-pro-002", "NMT"),
            index=0
        )
    # Layout with columns
    # "gemini-1.5-flash-001", "gemini-1.5-pro-001", "gemini-1.0-pro-001"
    b1, b2, b3 = st.columns([1, 0.5, 1])

    with b1:
        source = st.selectbox(
            "Source Language",
            keys_lang,
            index=48,
            key="source_lang"
        )
    with b3:
        target = st.selectbox(
            "Target Language",
            keys_lang,
            index=7,
            key="target_lang"
        )

    on = st.toggle("Text File")

    if on:
        uploaded_file = st.file_uploader(" ")
        if uploaded_file is not None:
            # To read file as bytes:
            #bytes_data = uploaded_file.getvalue()
            #st.write(bytes_data)
            filename = uploaded_file.name
            # To convert to a string based IO:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            #st.write(stringio)

            # To read file as string:
            string_data = stringio.read()
            #st.write()
            if st.button("Translate"):
                if llm_model == "NMT":
                    contents = [string_data]
                    translated_data = f"{generate_NMT(contents, languages[source], languages[target])[0]}"
                    st.download_button(label="Download Translated File", data = translated_data, file_name = 'Translated_file.txt')
                else:
                    translated_data = generate(string_data, languages[source], languages[target], llm_model)
                    st.download_button(label="Download Translated File", data = translated_data, file_name = 'Translated_file.txt')
        
                
            
                    
            # Can be used wherever a "file-like" object is accepted:
            #dataframe = pd.read_csv(uploaded_file)
            #st.write(dataframe)
            
    else:

        if "messages" not in st.session_state:
            st.session_state.messages = []



        # Chat input box
        prompt = st.chat_input("Type a text you want to translate")
        if prompt:
            # Save user input to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display status while processing
            with st.status("Translating...", expanded=True) as status:
                # Simulated delay to mimic processing (replace with actual call)
                #time.sleep(2)  # Replace with the time your `generate` function takes
                if llm_model == "NMT":
                    contents = [prompt]
                    response = f"{generate_NMT(contents, languages[source], languages[target])[0]}"
                else:
                    response = f"{generate(prompt, languages[source], languages[target], llm_model)}"    # Replace with your `generate` function
                st.session_state.messages.append({"role": "assistant", "content": response})
                status.update(label="Translated", state="complete", expanded=True)

        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

        c1, c3 = st.columns([0.1, 0.4])
        with c1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []

        with c3: 
            if st.button("Download Chat (.xlsx)"):
                user_messages = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
                assistant_messages = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]

                data = {"User": user_messages, "AI Response": assistant_messages}
                df_chat = pd.DataFrame(data)

                output_file = "chat_log.xlsx"
                df_chat.to_excel(output_file, index=False)

