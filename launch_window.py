import sys
import subprocess
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Home")
        self.setGeometry(100, 100, 400, 300)

        # Create a vertical layout to center the buttons.
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # heading = QLabel("Welcome to the App", self)
        # heading.setFont(QFont("Arial", 24, QFont.Weight.Bold)) 
        # # heading.setGeometry(150, 50, 300, 50)  # Position and size
        # heading.setStyleSheet("color: blue;")  
        # heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.button_app = QPushButton("Open Chat Interface")
        self.button_note = QPushButton("Open Note Interface")
        
        #a minimum size for the buttons
        self.button_app.setMinimumSize(200, 100)
        self.button_note.setMinimumSize(200, 100)
        
        
        layout.addWidget(self.button_app)
        layout.addWidget(self.button_note)
        
        # Connect button clicks to the corresponding methods.
        self.button_app.clicked.connect(self.open_app_interface)
        self.button_note.clicked.connect(self.open_note_interface)
        self.apply_styling()

    def open_app_interface(self):
        subprocess.Popen([sys.executable, "app.py"])  #launches the multi LLM

    def open_note_interface(self):
        subprocess.Popen([sys.executable, "note.py"])  #opens note taking interface

    def apply_styling(self):  
        with open("theme.qss", "r") as f:
            self.setStyleSheet(f.read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
