import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageDraw, ImageOps, ImageTk, ImageEnhance, ImageFilter
import numpy as np
from src.neural_net import SimpleNN

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("پروژه ۱۰: نسخه نهایی (Skeletonize Logic)")
        self.root.geometry("500x750")
        
        self.nn = SimpleNN(hidden_size=512) 
        try:
            self.nn.load_model('my_model.npz')
            print("Brain loaded successfully!")
        except:
            print("Error: Model not found. Please run main.py first!")

        self.brush_size = 20
        self.old_x = None
        self.old_y = None
        self.tk_image_ref = None 

        # --- UI ---
        ctrl_frame = tk.Frame(root, pady=15)
        ctrl_frame.pack()
        tk.Label(ctrl_frame, text="ضخامت قلم:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.slider = tk.Scale(ctrl_frame, from_=10, to=40, orient=tk.HORIZONTAL, command=self.change_brush)
        self.slider.set(self.brush_size)
        self.slider.pack(side=tk.LEFT, padx=10)

        self.canvas_width = 300
        self.canvas_height = 300
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='black', cursor="crosshair")
        self.canvas.pack(pady=5)
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)

        # دکمه‌ها
        btn_frame = tk.Frame(root, pady=5)
        btn_frame.pack()
        self.btn_predict = tk.Button(btn_frame, text="تشخیص بده", command=self.predict_digit, 
                                     bg='#4CAF50', fg='black', font=("Arial", 14, "bold"), height=2, width=15)
        self.btn_predict.pack(side=tk.LEFT, padx=5)
        self.btn_clear = tk.Button(btn_frame, text="پاک کردن", command=self.clear_canvas, 
                                   font=("Arial", 12), height=2, width=15)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        upload_frame = tk.Frame(root, pady=5)
        upload_frame.pack()
        self.btn_upload = tk.Button(upload_frame, text="📷 آپلود تصویر (File)", command=self.upload_image, 
                                    bg='#2196F3', fg='black', font=("Arial", 12), height=1, width=32)
        self.btn_upload.pack()
        
        self.lbl_result = tk.Label(root, text="یک عدد بکشید...", font=("Helvetica", 20, "bold"), fg="#333")
        self.lbl_result.pack(pady=10)

        debug_frame = tk.LabelFrame(root, text="دید هوش مصنوعی (استخوان‌بندی شده)", padx=10, pady=10)
        debug_frame.pack(pady=5)
        self.lbl_debug_img = tk.Label(debug_frame, text="[خالی]")
        self.lbl_debug_img.pack()

    def change_brush(self, val):
        self.brush_size = int(val)

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=self.brush_size, fill='white',
                                    capstyle=tk.ROUND, smooth=True, splinesteps=36)
            
            fat_brush = self.brush_size + 10 
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                           fill=255, width=fat_brush, joint="curve")
        self.old_x = event.x
        self.old_y = event.y

    def reset_coords(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="...", fg="#333")
        self.lbl_debug_img.config(image='', text="[خالی]")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path: return

        try:
            uploaded_img = Image.open(file_path).convert('L')
            
            # 1. کنتراست شدید (سفیدها سفیدتر، سیاه‌ها سیاه‌تر)
            enhancer = ImageEnhance.Contrast(uploaded_img)
            uploaded_img = enhancer.enhance(5.0)

            # 2. تشخیص و معکوس‌سازی (کاغذ سفید)
            avg_color = np.mean(np.array(uploaded_img))
            if avg_color > 100: 
                uploaded_img = ImageOps.invert(uploaded_img)
            
            # 3. حذف نویز پس‌زمینه (Threshold)
            uploaded_img = uploaded_img.point(lambda p: 255 if p > 150 else 0)

            # 4. *** تغییر اصلی: لاغر کردن قبل از کوچک کردن ***
            # عکس هنوز بزرگه (مثلا 1000 پیکسل). اینجا سایش میدیم.
            # 15 بار سایش میدیم تا فقط اسکلت بمونه!
            for _ in range(9): 
                uploaded_img = uploaded_img.filter(ImageFilter.MinFilter(3))

            # حالا که لاغر شد، می‌ندازیمش توی بوم
            uploaded_img.thumbnail((300, 300))
            self.clear_canvas()
            
            paste_x = (300 - uploaded_img.width) // 2
            paste_y = (300 - uploaded_img.height) // 2
            
            self.image.paste(uploaded_img, (paste_x, paste_y))
            
            self.tk_image_ref = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.tk_image_ref, anchor="nw")
            
            self.lbl_result.config(text="تصویر بارگذاری و لاغر شد.", fg="blue")

        except Exception as e:
            print(f"Error: {e}")

    def center_image_by_mass(self, img):
        img_array = np.array(img)
        y_idxs, x_idxs = np.nonzero(img_array)
        if len(y_idxs) == 0: return img
        com_y = np.mean(y_idxs)
        com_x = np.mean(x_idxs)
        shift_y = 14 - com_y
        shift_x = 14 - com_x
        return img.transform(img.size, Image.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y))

    def predict_digit(self):
        bbox = self.image.getbbox()
        if bbox is None: return

        cropped = self.image.crop(bbox)
        
        width, height = cropped.size
        max_dim = max(width, height)
        
        # زوم: 20 پیکسل (متعادل)
        ratio = 20.0 / max_dim 
        new_size = (int(width * ratio), int(height * ratio))
        img_resized = cropped.resize(new_size, Image.Resampling.LANCZOS)
        
        temp_img = Image.new("L", (28, 28), 0)
        paste_x = (28 - new_size[0]) // 2
        paste_y = (28 - new_size[1]) // 2
        temp_img.paste(img_resized, (paste_x, paste_y))
        
        final_img = self.center_image_by_mass(temp_img)

        # افزایش کنتراست نهایی
        enhancer = ImageEnhance.Contrast(final_img)
        final_img = enhancer.enhance(2.0)

        debug_view = final_img.resize((112, 112), Image.Resampling.NEAREST)
        debug_photo = ImageTk.PhotoImage(debug_view)
        self.lbl_debug_img.config(image=debug_photo, text="")
        self.lbl_debug_img.image = debug_photo

        img_array = np.array(final_img)
        img_vector = img_array.reshape(1, 784).astype(np.float32) / 255.0
        
        probs = self.nn.forward(img_vector)
        prediction = np.argmax(probs)
        confidence = np.max(probs) * 100
        
        color = "#008000" if confidence > 80 else "#FF8C00"
        self.lbl_result.config(text=f"تشخیص: {prediction} ({confidence:.1f}%)", fg=color)

if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = DigitRecognizerApp(root)
    root.mainloop()