import os
from pyshortcuts import make_shortcut


root = os.path.dirname(os.path.join(__file__))
app = os.path.join(root,'main.py')
icon = os.path.join(root,'pancake.ico')

make_shortcut(app, name='MIDAS', icon=icon)
print(f"MIDAS Desktop shortcut is created by { os.getlogin()}. ")