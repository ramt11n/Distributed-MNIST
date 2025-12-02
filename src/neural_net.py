import numpy as np

class SimpleNN:
    def __init__(self, input_size=784, hidden_size=512, output_size=10):
        # مقداردهی اولیه وزن‌ها (تصادفی)
        # لایه اول (وزن‌ها و بایاس‌ها)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        # لایه دوم (خروجی)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        # برای تبدیل خروجی‌ها به احتمالات (جمعشون بشه ۱)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """حرکت رو به جلو: پیش‌بینی بر اساس ورودی"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)  # خروجی لایه مخفی
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = self.softmax(self.z2)  # پیش‌بینی نهایی
        return self.probs

    def get_gradients(self, X, y_one_hot):
        """
        قلب تپنده MapReduce:
        این تابع خطا را حساب کرده و گرادیان‌ها (جهت اصلاح) را برمی‌گرداند.
        """
        m = X.shape[0] # تعداد نمونه‌ها در این بچ
        
        # 1. انجام پیش‌بینی
        probs = self.forward(X)
        
        # 2. محاسبه خطا (Backward Pass)
        # مشتق لایه آخر: (پیش‌بینی - واقعیت)
        delta2 = probs - y_one_hot
        
        # گرادیان‌های لایه دوم
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # مشتق لایه مخفی (زنجیره‌ای)
        delta1 = np.dot(delta2, self.W2.T) * (self.a1 * (1 - self.a1))
        
        # گرادیان‌های لایه اول
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def update_weights(self, grads, learning_rate=0.1):
        """آپدیت کردن وزن‌ها بر اساس گرادیان (مرحله Reduce)"""
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']

    def get_accuracy(self, X, y):
        # محاسبه دقت مدل
        predictions = np.argmax(self.forward(X), axis=1)
        # اگر y به صورت one-hot نیست، خودش را استفاده کن
        if len(y.shape) > 1: 
             y = np.argmax(y, axis=1)
        return np.mean(predictions == y)
    
    #codes for pillow (UI):
    def save_model(self, filename='model.npz'):
        """ذخیره وزن‌ها در فایل"""
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Model saved to {filename}")

    def load_model(self, filename='model.npz'):
        """لود کردن وزن‌ها از فایل"""
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Model loaded from {filename}")