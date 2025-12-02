import gzip
import struct
import numpy as np
import os

def load_images(filename):
    # باز کردن فایل فشرده تصاویر
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError("Magic number mismatch in image file!")
        
        # خواندن کل دیتا
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        
        # تغییر شکل: (تعداد عکس, ۲۸, ۲۸)
        data = data.reshape(num, rows, cols)
        
        # نرمال‌سازی (اعداد بین ۰ و ۱ بشن) و فلت کردن (تبدیل به بردار خطی)
        # هر عکس ۲۸*۲۸ میشه یه خط ۷۸۴ تایی
        return (data.reshape(num, rows * cols) / 255.0).astype(np.float32)

def load_labels(filename):
    # باز کردن فایل فشرده لیبل‌ها
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError("Magic number mismatch in label file!")
        
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_data(data_folder='./data'):
    # مسیر فایل‌ها رو اینجا می‌سازیم
    files = {
        'train_img': os.path.join(data_folder, 'train-images-idx3-ubyte.gz'),
        'train_lbl': os.path.join(data_folder, 'train-labels-idx1-ubyte.gz'),
        'test_img': os.path.join(data_folder, 't10k-images-idx3-ubyte.gz'),
        'test_lbl': os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz')
    }

    # لود کردن همه
    print("Loading MNIST data...")
    X_train = load_images(files['train_img'])
    y_train = load_labels(files['train_lbl'])
    X_test = load_images(files['test_img'])
    y_test = load_labels(files['test_lbl'])

    print(f"Data Loaded! Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test