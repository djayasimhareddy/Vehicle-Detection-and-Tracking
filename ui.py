from tkinter import filedialog, Tk, Button, Label

def create_gui_menu():
    root = Tk()
    root.title("Vehicle Detection")
    root.geometry("400x300")
    root.config(bg="#2c3e50")

    def webcam(): root.choice, root.path = "webcam", None; root.destroy()
    def video():
        p = filedialog.askopenfilename(title="Video", filetypes=[("Video", "*.mp4 *.webm"), ("All", "*.*")])
        if p: root.choice, root.path = "video", p; root.destroy()
    def image():
        p = filedialog.askopenfilename(title="Image", filetypes=[("Image", "*.jpg *.png"), ("All", "*.*")])
        if p: root.choice, root.path = "image", p; root.destroy()

    st = ("Arial", 12, "bold")
    Label(root, text="Select Input Mode", font=("Arial", 18, "bold"), fg="white", bg="#2c3e50").pack(pady=20)
    Button(root, text="📷 Webcam", command=webcam, font=st, width=20, height=2).pack(pady=10)
    Button(root, text="🎬 Video", command=video, font=st, width=20, height=2).pack(pady=10)
    Button(root, text="🖼️ Image", command=image, font=st, width=20, height=2).pack(pady=10)

    root.choice, root.path = None, None
    root.mainloop()
    return root.choice, root.path
