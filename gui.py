import customtkinter
import text_to_image as tig
from PIL import Image
import threading as th

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("800x900")
save_path = []
filename = []
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)
    
def generate_image(prompt, filename, save_path):
    filename.append(tig.generate_image_from_text(prompt, save_path))

def save_directory_path():
    dir = customtkinter.filedialog.askdirectory()
    while dir=='':
        txtPrompt.forget()
        btnGenImage.forget()
        dir = customtkinter.filedialog.askdirectory()
    else:
        save_path.append(dir)        
        txtPrompt.pack(pady=12, padx=10)
        btnGenImage.pack(pady=12, padx=10)
        

def gen_image():
    lblLoading.configure(text="Please wait. The image is being generated...")
    lblImg.configure(image=None)
    lblLoading.pack(pady=12, padx=10)
    prompt = txtPrompt.get("1.0", "end") if txtPrompt.get("1.0", "end")!="" else "1girl, fluffy red hair, cute face, sunny beach, best quality, normal hands, illustration, contour deepening, detailed glow Steps: 20, Sampler: DPM++  2S a, CFG scale: 8, Seed: 4186044705, Size: 704x896"
    generate_image(prompt, filename, save_path[0])
       
    if filename==['']:
        lblLoading.configure(text="Potential NSFW image. Please change the prompt.")    
    else:
        lblLoading.configure(text="Image generated successfully!!")
        img = customtkinter.CTkImage(Image.open(r''+filename[0]).resize((512,512)),size=(704, 896))   
        lblImg.configure(image=img)             
        lblImg.pack(pady=12, padx=10)

lblTitle = customtkinter.CTkLabel(master=frame, text="Text-to-Image", font=("Roboto", 24))
lblTitle.pack(pady=12, padx=10)

lblDir = customtkinter.CTkLabel(master=frame, text="Select Directory to store generated image.")
lblDir.pack(pady=12, padx=10)

btnSaveDir = customtkinter.CTkButton(master=frame, text="Click here to select directory", command=save_directory_path,)
btnSaveDir.pack(pady=12, padx=10)

txtPrompt = customtkinter.CTkTextbox(master=frame, height=120, width=280)

btnGenImage = customtkinter.CTkButton(master=frame, text="Generate Image", command=gen_image, )

lblLoading = customtkinter.CTkLabel(master=frame, text="Please wait. The image is being generated...")

lblImg = customtkinter.CTkLabel(master=frame, text="")


root.mainloop()