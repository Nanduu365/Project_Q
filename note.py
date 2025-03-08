import sys
import json
import time
import pyaudio
from vosk import Model, KaldiRecognizer
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal

class SpeechThread(QThread):
    # Emit recognized text along with a flag indicating whether the result is final.
    result_signal = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        
        #Loading the vosk model
        self.model = Model("vosk_model")
        self.recognizer = None
        self.audio_interface = pyaudio.PyAudio()

    def run(self):
        self.running = True
        # Opening an audio stream: mono, 16-bit, 16000 Hz, and use a suitable buffer size.
        stream = self.audio_interface.open(format=pyaudio.paInt16,
                                           channels=1,
                                           rate=16000,
                                           input=True,
                                           frames_per_buffer=8000)
        stream.start_stream()
        self.recognizer = KaldiRecognizer(self.model, 16000)

        while self.running:
            data = stream.read(4000, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                # Final result when the recognizer finalizes a phrase.
                result = self.recognizer.Result()
                result_dict = json.loads(result)
                text = result_dict.get("text", "")
                self.result_signal.emit(text, True)
            else:
                # Partial result for real-time (incomplete) feedback.
                partial_result = self.recognizer.PartialResult()
                result_dict = json.loads(partial_result)
                partial_text = result_dict.get("partial", "")
                self.result_signal.emit(partial_text, False)
            # Delay slightly to pace the updates  to human speech.
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        self.audio_interface.terminate()

    def stop(self):
        self.running = False
        self.wait()  # Wait for the thread to finish cleanly

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat Interface")
        self.setGeometry(100, 100, 600, 400)

        # two text displays:
        # 1. current_display: shows the current (partial) recognized text.
        # 2. cumulative_display: keeps appending finalized texts.
        self.current_display = QTextEdit()
        self.current_display.setReadOnly(True)
        self.cumulative_display = QTextEdit()
        self.cumulative_display.setReadOnly(False)

        # Layout for the text displays.
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.current_display)
        main_layout.addWidget(self.cumulative_display)

        # Create a horizontal layout for the 4 buttons.
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.clear_button = QPushButton("Clear")
        self.download_button = QPushButton("Download")
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.download_button)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        
        self.start_button.clicked.connect(self.start_recognition)
        self.pause_button.clicked.connect(self.pause_recognition)
        self.clear_button.clicked.connect(self.clear_data)
        self.download_button.clicked.connect(self.download_data)

        # Speech thread is initialized when recognition starts
        self.speech_thread = None

        self.apply_styling()

    def start_recognition(self):
        if self.speech_thread is None or not self.speech_thread.isRunning():
            self.speech_thread = SpeechThread()
            self.speech_thread.result_signal.connect(self.update_text)
            self.speech_thread.start()

    def pause_recognition(self):
        if self.speech_thread is not None and self.speech_thread.isRunning():
            self.speech_thread.stop()
            self.speech_thread = None

    def update_text(self, text, is_final):
        if not is_final:
            # For partial results, update the current text display.
            self.current_display.setPlainText(text)
        else:
            # For final results, append the recognized text to the cumulative display
            # and then clear the current display.
            if text.strip():
                current_transcript = self.cumulative_display.toPlainText()
                if current_transcript:
                    new_transcript = current_transcript + "\n" + text.strip()
                else:
                    new_transcript = text.strip()
                self.cumulative_display.setPlainText(new_transcript)
            self.current_display.clear()

    def clear_data(self):
        # Clear both displays.
        self.current_display.clear()
        self.cumulative_display.clear()

    def download_data(self):
        # as a .txt file.
        filename, _ = QFileDialog.getSaveFileName(self, "Save Transcript", "", "Text Files (*.txt)")
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.cumulative_display.toPlainText())

    def apply_styling(self):
        with open("theme.qss", "r") as f:
            self.setStyleSheet(f.read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
