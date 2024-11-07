# main.py

import tkinter as tk
from mlp.gui import MLPApp

def main():
    root = tk.Tk()
    app = MLPApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
