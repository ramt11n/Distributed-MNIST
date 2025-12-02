import numpy as np
import multiprocessing
import time
from src.mnist_loader import load_data
from src.neural_net import SimpleNN
from src.worker import worker_task

# تنظیمات
# تنظیمات
EPOCHS = 50
BATCH_SIZE = 128        
# ...     
NUM_WORKERS = 4         
LEARNING_RATE = 0.05

def to_one_hot(y, num_classes=10):
    """تبدیل اعداد ساده (مثل 5) به بردار (مثل [0,0,0,0,0,1,0,0,0,0])"""
    return np.eye(num_classes)[y]

def create_chunks(X, y, num_chunks):
    """داده‌ها را به تکه‌های مساوی برای کارگرها تقسیم می‌کند"""
    chunk_size = len(X) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_chunks - 1 else len(X)
        chunks.append((X[start:end], y[start:end]))
    return chunks

def average_gradients(grads_list):
    """مرحله Reduce: میانگین‌گیری از نتایج کارگرها"""
    avg_grads = {}
    keys = grads_list[0].keys()
    for key in keys:
        avg_grads[key] = np.mean([g[key] for g in grads_list], axis=0)
    return avg_grads

if __name__ == "__main__":
    # 1. لود کردن داده‌ها
    print("--- [Step 1] Loading Data ---")
    X_train, y_train, X_test, y_test = load_data('./data')
    
    # *** اصلاح مهم: تبدیل لیبل‌های آموزشی به فرمت One-Hot برای شبکه عصبی ***
    y_train_encoded = to_one_hot(y_train)

    # 2. ساخت مدل اصلی (رئیس)
    print("--- [Step 2] Initializing Neural Network ---")
    master_nn = SimpleNN()
    
    # 3. آماده‌سازی استخر پردازش
    # نکته: در مک ممکن است نیاز باشد متد شروع را روی fork تنظیم کنیم
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass # اگر قبلاً ست شده بود مشکلی نیست

    pool = multiprocessing.Pool(processes=NUM_WORKERS)
    print(f"--- [Step 3] MapReduce Engine Started with {NUM_WORKERS} Workers ---")

    start_time = time.time()

    # 4. حلقه آموزش (Training Loop)
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # بر هم زدن داده‌ها (Shuffle)
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        y_shuffled_encoded = y_train_encoded[permutation] # نسخه One-Hot را شافل می‌کنیم

        # پردازش دسته به دسته
        num_batches = len(X_train) // BATCH_SIZE
        
        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled_encoded[start:end]

            # --- فاز MAP ---
            chunks = create_chunks(X_batch, y_batch, NUM_WORKERS)
            
            current_weights = {
                'W1': master_nn.W1, 'b1': master_nn.b1,
                'W2': master_nn.W2, 'b2': master_nn.b2
            }
            tasks = [(current_weights, ch[0], ch[1]) for ch in chunks]

            # ارسال به کارگرها
            gradients_list = pool.map(worker_task, tasks)

            # --- فاز REDUCE ---
            final_grads = average_gradients(gradients_list)
            master_nn.update_weights(final_grads, LEARNING_RATE)

        # محاسبه دقت (از y_test اصلی استفاده می‌کنیم که عدد است)
        accuracy = master_nn.get_accuracy(X_test, y_test)
        duration = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{EPOCHS} | Accuracy: {accuracy*100:.2f}% | Time: {duration:.2f}s")

    total_time = time.time() - start_time
    print("\n" + "="*40)
    print(f"TRAINING FINISHED in {total_time:.2f} seconds.")
    print(f"Final Model Accuracy: {master_nn.get_accuracy(X_test, y_test)*100:.2f}%")
    print("="*40)

    pool.close()
    pool.join()

    #Pillow (UI):
    # ... کدهای قبلی ...
    print(f"Final Model Accuracy: {master_nn.get_accuracy(X_test, y_test)*100:.2f}%")
    print("="*40)
    
    # اضافه شده: ذخیره مدل
    master_nn.save_model('my_model.npz') # <--- این خط جدید است

    pool.close()
    pool.join()