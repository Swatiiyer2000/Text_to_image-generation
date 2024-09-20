import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
import tkinter as tk
import customtkinter as ctk

# Create the app
app = ctk.CTk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)
prompt.pack()

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

# Model and device setup
model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Set device to CPU

# Load model with CPU dtype
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32  # For CPU, use float32
)
pipe.to(device)

def generate():
    prompt_text = prompt.get()

    with torch.no_grad():  # Disable gradients for inference
        image = pipe(prompt_text, guidance_scale=8.5).images[0]
    
    # Save or display image
    image.save('generated_image.png')

    # Convert the image to a format compatible with CTkImage
    pil_image = Image.open('generated_image.png')
    ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(512, 512))  # Convert to CTkImage
    
    # Use the CTkImage in the label
    lmain.configure(image=ctk_image)
    lmain.image = ctk_image  # Keep reference to avoid garbage collection

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
