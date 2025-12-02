import numpy as np
from src.neural_net import SimpleNN

def worker_task(payload):
    """
    این تابع روی یک پروسه جداگانه (هسته دیگر CPU) اجرا می‌شود.
    
    ورودی (payload) یک تاپل است شامل:
    1. weights: دیکشنری وزن‌های فعلی مدل (که رئیس فرستاده)
    2. X_chunk: تکه‌ای از تصاویر برای پردازش
    3. y_chunk: لیبل‌های مربوط به آن تصاویر
    """
    weights, X_chunk, y_chunk = payload

    # 1. ساخت یک نمونه محلی از شبکه عصبی
    # (چون هر کارگر حافظه خودش را دارد)
    nn = SimpleNN()
    
    # 2. همگام‌سازی مغز کارگر با مغز رئیس
    # وزن‌های دریافتی را جایگزین وزن‌های تصادفی می‌کنیم
    nn.W1 = weights['W1']
    nn.b1 = weights['b1']
    nn.W2 = weights['W2']
    nn.b2 = weights['b2']

    # 3. انجام محاسبات سنگین (Map Step)
    # محاسبه گرادیان روی این تکه داده
    gradients = nn.get_gradients(X_chunk, y_chunk)

    # 4. بازگرداندن نتیجه به رئیس
    return gradients