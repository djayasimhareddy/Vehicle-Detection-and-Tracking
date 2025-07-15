# gui.py
import os
from tkinter import Tk, Label, Button, filedialog
from config import CLASS_NAMES

def create_gui_menu():
    root = Tk()
    root.title("Vehicle Detection & Counting System")
    root.geometry("500x400")
    root.configure(bg='#2c3e50')

    def choose_webcam():
        root.choice, root.path = 'webcam', None
        root.quit()

    def choose_video():
        base = os.path.dirname(os.path.abspath(__file__))
        initial = os.path.join(base, "data", "videos")
        p = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=initial if os.path.exists(initial) else None,
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")]
        )
        if p:
            root.choice, root.path = 'video', p
            root.quit()

    def choose_image():
        base = os.path.dirname(os.path.abspath(__file__))
        initial = os.path.join(base, "data", "images")
        p = filedialog.askopenfilename(
            title="Select Image File",
            initialdir=initial if os.path.exists(initial) else None,
            filetypes=[("Image","*.jpg *.jpeg *.png"),("All","*.*")]
        )
        if p:
            root.choice, root.path = 'image', p
            root.quit()

    Label(root, text="Vehicle Detection & Counting", 
          font=('Arial',18,'bold'), fg='white', bg='#2c3e50').pack(pady=20)
    Label(root, text="Classes: " + ", ".join(CLASS_NAMES.values()),
          font=('Arial',10), fg='#bdc3c7', bg='#2c3e50').pack(pady=5)

    btn_cfg = {'font':('Arial',12,'bold'),'width':20,'height':2,'bd':0,'cursor':'hand2'}
    Button(root, text="📹 Webcam", command=choose_webcam, bg='#3498db', fg='white', **btn_cfg).pack(pady=10)
    Button(root, text="🎬 Video",  command=choose_video, bg='#e74c3c', fg='white', **btn_cfg).pack(pady=10)
    Button(root, text="🖼️ Image",  command=choose_image, bg='#f39c12', fg='white', **btn_cfg).pack(pady=10)
    Button(root, text="❌ Exit",   command=root.destroy, bg='#95a5a6', fg='white', **btn_cfg).pack(pady=10)

    Label(root, text="Controls: 'q'/ESC=quit, 's'=save", 
          font=('Arial',9), fg='#bdc3c7', bg='#2c3e50').pack(pady=20)

    root.choice, root.path = None, None
    root.mainloop()
    return root.choice, root.path

def console_menu():
    print("1) Webcam\n2) Video\n3) Image\n4) Exit")
    while True:
        c = input("Choice: ").strip()
        if c == '1':
            return 'webcam', None
        if c == '2':
            p = input("Video path: ").strip()
            if os.path.exists(p):
                return 'video', p
        if c == '3':
            p = input("Image path: ").strip()
            if os.path.exists(p):
                return 'image', p
        if c == '4':
            return 'exit', None


