from PyQt5.QtWidgets import *

def show_message(program, message):
    QMessageBox.information(program, "Alert", message)