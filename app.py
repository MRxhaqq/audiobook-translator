import streamlit as st
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import io
import os
import tempfile
from pydub import AudioSegment
from pydub.playback import play
import base64

# Configure the page
st.set_page_config(
    page_title="Language Audiobook Translator",
    page_icon="🎧",
    layout="wide"
)

# Title and description
st.title("🎧 Language Audiobook Translator")
st.markdown("Convert audiobooks from one language to another using speech recognition and translation.")


# Initialize components with better error handling
@st.cache_resource
def init_components():
    recognizer = sr.Recognizer()
    # Don't cache the translator to avoid session issues
    return recognizer


recognizer = init_components()


# Create translator instances as needed to avoid caching issues
def get_fresh_translator():
    """Get a fresh translator instance to avoid async issues."""
    return Translator()


# Language options
LANGUAGES = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese (Simplified)': 'zh-cn',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Dutch': 'nl',
    'Swedish': 'sv',
    'Norwegian': 'no'
}


def convert_audio_format(audio_file):
    """Convert audio file to WAV format for speech recognition with improved settings."""
    try:
        # First, try to use the file directly if it's already WAV
        if audio_file.name.lower().endswith('.wav'):
            return audio_file

        # Try using pydub with FFmpeg
        audio = AudioSegment.from_file(audio_file)

        # Improve audio for speech recognition
        # Normalize audio level
        audio = audio.normalize()

        # Ensure mono audio (speech recognition works better with mono)
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Set sample rate to 16kHz (optimal for speech recognition)
        audio = audio.set_frame_rate(16000)

        # Export to WAV with better settings
        wav_io = io.BytesIO()
        audio.export(
            wav_io,
            format="wav",
            parameters=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz
        )
        wav_io.seek(0)

        return wav_io

    except FileNotFoundError as e:
        # FFmpeg not found - provide user-friendly error with solutions
        st.error("""
        🚨 **FFmpeg Not Found!**

        **Quick Fixes (choose one):**

        1️⃣ **Easiest - Use WAV files:**
           - Convert your audio to WAV format first
           - Use VLC Player: Media → Convert/Save
           - Or use online converter: convertio.co

        2️⃣ **Install FFmpeg:**
           ```
           # In Command Prompt as Administrator:
           winget install FFmpeg
           ```

        3️⃣ **Alternative install:**
           ```
           pip install imageio[ffmpeg]
           ```
        """)
        return None

    except Exception as e:
        # For other errors, try alternative approach
        st.warning(f"Audio conversion issue: {str(e)}")
        st.info("💡 **Try uploading a WAV file instead** - no conversion needed!")
        return None


def transcribe_audio(audio_data, source_lang):
    """Convert speech to text with improved settings."""
    try:
        with sr.AudioFile(audio_data) as source:
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Record the entire audio file
            audio = recognizer.record(source, duration=None)  # Record everything

        # Try Google Speech Recognition with better settings
        try:
            text = recognizer.recognize_google(
                audio,
                language=source_lang,
                show_all=False  # Get best result only
            )
            return text
        except sr.UnknownValueError:
            # Try with different recognition settings
            try:
                # Try again with show_all=True to get alternatives
                results = recognizer.recognize_google(
                    audio,
                    language=source_lang,
                    show_all=True
                )
                if results and 'alternative' in results:
                    # Get the best alternative
                    best_result = results['alternative'][0]
                    if 'transcript' in best_result:
                        return best_result['transcript']
                return "Could not understand audio - try with clearer audio"
            except:
                return "Speech recognition failed - audio may be unclear"

    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Transcription error: {e}"


def translate_text(text, source_lang, target_lang):
    """Translate text from source to target language."""
    try:
        # First try the alternative method (more reliable)
        return translate_text_alternative(text, source_lang, target_lang)
    except:
        # If alternative fails, try googletrans with proper handling
        try:
            import time
            time.sleep(0.5)  # Brief delay
            fresh_translator = Translator()
            translation = fresh_translator.translate(text, src=source_lang, dest=target_lang)

            # Check if it's a coroutine (async object)
            if hasattr(translation, '__await__'):
                return "Error: Received async object - please install deep_translator"

            # Check if it has text attribute
            if hasattr(translation, 'text'):
                result = translation.text
                # Ensure it's not a coroutine string representation
                if 'coroutine' in str(result).lower():
                    return "Error: Translation returned coroutine object"
                return result
            else:
                return "Error: Translation object missing text attribute"

        except Exception as e:
            return f"Translation error: {str(e)}"


def translate_text_alternative(text, source_lang, target_lang):
    """Alternative translation using deep_translator - more reliable."""
    try:
        from deep_translator import GoogleTranslator

        # Handle language code differences
        lang_mapping = {
            'zh-cn': 'zh',
            'zh': 'zh-cn'
        }

        source_mapped = lang_mapping.get(source_lang, source_lang)
        target_mapped = lang_mapping.get(target_lang, target_lang)

        translator = GoogleTranslator(source=source_mapped, target=target_mapped)
        result = translator.translate(text)

        # Ensure we got a valid string result
        if result and isinstance(result, str) and 'coroutine' not in result.lower():
            return result
        else:
            return f"Alternative translation failed: Invalid result format"

    except ImportError:
        # Try a simple requests-based approach as final fallback
        try:
            import requests
            import urllib.parse

            # Simple Google Translate API call
            base_url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': source_lang,
                'tl': target_lang,
                'dt': 't',
                'q': text
            }

            response = requests.get(base_url, params=params, timeout=10)
            result = response.json()

            if result and len(result) > 0 and len(result[0]) > 0:
                return result[0][0][0]
            else:
                return "Translation failed: Invalid API response"

        except Exception as e:
            return f"All translation methods failed. Please install: pip install deep_translator"

    except Exception as e:
        return f"Alternative translation error: {str(e)}"


def text_to_speech(text, lang):
    """Convert text to speech and return audio data."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)

        # Save to BytesIO object
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return audio_buffer
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None


def get_audio_download_link(audio_buffer, filename):
    """Generate download link for audio file."""
    audio_bytes = audio_buffer.getvalue()
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Translated Audiobook</a>'
    return href


# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎤 Input Settings")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your audiobook file",
        type=['mp3', 'wav', 'ogg', 'flac', 'm4a'],
        help="Supported formats: MP3, WAV, OGG, FLAC, M4A"
    )

    # Source language
    source_language = st.selectbox(
        "Source Language (Original audiobook language)",
        options=list(LANGUAGES.keys()),
        index=0
    )

with col2:
    st.subheader("🎯 Output Settings")

    # Target language
    target_language = st.selectbox(
        "Target Language (Translate to)",
        options=list(LANGUAGES.keys()),
        index=1
    )

    # Processing options
    chunk_size = st.slider(
        "Audio chunk size (seconds)",
        min_value=10,
        max_value=60,
        value=30,
        help="Larger chunks may be more accurate but take longer to process"
    )

# Processing section
if uploaded_file is not None:
    st.subheader("🔄 Processing")

    # Display file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
        "File type": uploaded_file.type
    }
    st.json(file_details)

    # Process button
    if st.button("🚀 Start Translation Process", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Convert audio format
            status_text.text("Converting audio format...")
            progress_bar.progress(10)

            wav_audio = convert_audio_format(uploaded_file)
            if wav_audio is None:
                st.error("Failed to convert audio format")
                st.stop()

            # Step 2: Transcribe audio
            status_text.text("Converting speech to text...")
            progress_bar.progress(30)

            source_lang_code = LANGUAGES[source_language]
            transcribed_text = transcribe_audio(wav_audio, source_lang_code)

            if "error" in transcribed_text.lower() or "could not understand" in transcribed_text.lower():
                st.error(f"Transcription failed: {transcribed_text}")
                st.stop()

            # Display transcribed text with audio info
            st.subheader("📝 Transcribed Text")

            # Show audio file info
            if wav_audio:
                try:
                    from pydub import AudioSegment

                    if hasattr(wav_audio, 'seek'):
                        wav_audio.seek(0)  # Reset position
                        temp_audio = AudioSegment.from_wav(wav_audio)
                        duration = len(temp_audio) / 1000.0  # Duration in seconds
                        sample_rate = temp_audio.frame_rate
                        st.info(f"🎵 Audio Info: {duration:.1f} seconds, {sample_rate} Hz sample rate")
                except:
                    pass

            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_area("Original text:", transcribed_text, height=100)
            with col2:
                st.metric("Characters", len(transcribed_text))

            # Check if transcription seems incomplete
            if len(transcribed_text) < 50:
                st.warning("⚠️ **Short transcription detected!** This might indicate:")
                st.write("- Audio quality issues")
                st.write("- Background noise")
                st.write("- Microphone too far from speaker")
                st.write("- Audio file corruption")

                if st.button("🔄 Try Again with Different Settings"):
                    st.rerun()

            # Step 3: Translate text
            status_text.text("Translating text...")
            progress_bar.progress(60)

            target_lang_code = LANGUAGES[target_language]
            translated_text = translate_text(transcribed_text, source_lang_code, target_lang_code)

            # If primary translation fails, try alternative method
            if "error" in translated_text.lower():
                status_text.text("Trying alternative translation method...")
                translated_text = translate_text_alternative(transcribed_text, source_lang_code, target_lang_code)

            if "error" in translated_text.lower() or "failed" in translated_text.lower():
                st.error(f"Translation failed: {translated_text}")
                st.info("💡 **Troubleshooting Tips:**")
                st.write("- Check your internet connection")
                st.write("- Try a shorter text sample")
                st.write("- Install alternative translator: `pip install deep_translator`")
                st.stop()

            # Display translated text
            st.subheader("🔄 Translated Text")
            st.text_area("Translated text:", translated_text, height=100)

            # Step 4: Convert to speech
            status_text.text("Converting text to speech...")
            progress_bar.progress(80)

            audio_buffer = text_to_speech(translated_text, target_lang_code)
            if audio_buffer is None:
                st.error("Failed to generate speech")
                st.stop()

            # Step 5: Complete
            status_text.text("Translation complete!")
            progress_bar.progress(100)

            # Display results
            st.subheader("🎵 Results")

            # Play audio
            st.audio(audio_buffer.getvalue(), format='audio/mp3')

            # Download link
            download_filename = f"translated_{uploaded_file.name.split('.')[0]}.mp3"
            st.markdown(get_audio_download_link(audio_buffer, download_filename), unsafe_allow_html=True)

            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Length", f"{len(transcribed_text)} chars")
            with col2:
                st.metric("Translated Length", f"{len(translated_text)} chars")
            with col3:
                st.metric("Translation", f"{source_language} → {target_language}")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            status_text.text("Processing failed")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown("""
    1. **Upload** your audiobook file
    2. **Select** source and target languages
    3. **Click** start translation
    4. **Download** the translated audiobook

    ---

    ### Supported Features:
    - 🎵 Multiple audio formats
    - 🌍 15+ languages
    - 🎧 High-quality speech synthesis
    - 📱 Mobile-friendly interface
    - 💾 Download translated files

    ---

    ### Tips for best results:
    - Use clear, high-quality audio
    - Avoid background noise
    - Choose appropriate chunk sizes
    - Ensure stable internet connection
    - Speak clearly and at normal pace
    - Use WAV format for best quality

    ### If transcription is incomplete:
    - Check audio file isn't corrupted
    - Try converting to WAV first
    - Reduce background noise
    - Ensure audio is loud enough
    """)

    st.header("Requirements")
    st.code("""
# Install required packages:
pip install streamlit
pip install SpeechRecognition
pip install googletrans==4.0.0rc1
pip install gTTS
pip install pydub

# For alternative translation (if needed):
pip install deep_translator
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎧 Language Audiobook Translator | Built with Streamlit</p>
        <p>Transform any audiobook into your preferred language!</p>
    </div>
    """,
    unsafe_allow_html=True

)
