import os
import time
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import mss
import torch
import random
import shutil
import datetime

RAW_DIR = 'raw_screens'
LABELS_DIR = 'labels'
DATASET_DIR = 'dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_YOLO_DIR = os.path.join(DATASET_DIR, 'labels')
CLASSES_TXT = os.path.join(DATASET_DIR, 'classes.txt')
DATASET_YAML = os.path.join(DATASET_DIR, 'dataset.yaml')
YOLOV5_CFG = 'yolov5s.yaml'
MODEL_PATH = 'runs/train/auto_yolo_train/weights/best.pt'
IMG_SIZE = 640
TRAIN_RATIO = 0.8
REFRESH_MS = 1

def collect_screenshots():
    os.makedirs(RAW_DIR, exist_ok=True)
    sct = mss.mss()
    monitor = sct.monitors[1]
    existing = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(RAW_DIR) if f.startswith('screenshot_') and f.endswith('.png')]
    idx = max(existing) + 1 if existing else 0
    print('Скриншоты начнут делаться через 5 секунд!')
    time.sleep(5)
    max_count = 250
    count = 0
    try:
        while count < max_count:
            img = sct.grab(monitor)
            im = Image.frombytes('RGB', img.size, img.rgb)
            fname = f'screenshot_{idx:04d}.png'
            im.save(os.path.join(RAW_DIR, fname))
            print(f'Сохранён {fname}')
            idx += 1
            count += 1
            time.sleep(1)
        print(f'Сделано {max_count} скриншотов, сбор завершён.')
    except KeyboardInterrupt:
        print('Сбор скриншотов остановлен.')

def label_screenshots():
    os.makedirs(LABELS_DIR, exist_ok=True)
    root_class = tk.Tk()
    root_class.withdraw()
    class_name = simpledialog.askstring('Класс', 'Введите имя класса для всех bbox:', parent=root_class)
    if not class_name:
        print('Класс не введён, разметка отменена.')
        return
    root_class.destroy()
    if os.path.exists('class_map.txt'):
        with open('class_map.txt', 'r', encoding='utf-8') as cmap:
            lines = cmap.readlines()
            class_map = {l.strip().split(':')[1]: int(l.strip().split(':')[0]) for l in lines if ':' in l}
    else:
        class_map = {}
    if class_name not in class_map:
        class_id = len(class_map)
        with open('class_map.txt', 'a', encoding='utf-8') as cmap:
            cmap.write(f'{class_id}:{class_name}\n')
    else:
        class_id = class_map[class_name]
    img_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.png')])
    for img_file in img_files:
        label_file = os.path.join(LABELS_DIR, img_file.replace('.png', '.txt'))
        if os.path.exists(label_file):
            continue
        img_path = os.path.join(RAW_DIR, img_file)
        img = Image.open(img_path)
        w, h = img.size
        bboxes = []
        root = tk.Tk()
        root.title(f'Разметка: {img_file}')
        tkimg = ImageTk.PhotoImage(img.resize((min(w, 1280), min(h, 720))))
        canvas = tk.Canvas(root, width=tkimg.width(), height=tkimg.height())
        canvas.pack()
        canvas.create_image(0, 0, anchor='nw', image=tkimg)
        rect = [None]
        start = [0, 0]
        boxes = []
        def on_down(event):
            start[0], start[1] = event.x, event.y
            rect[0] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red')
        def on_drag(event):
            if rect[0]:
                canvas.coords(rect[0], start[0], start[1], event.x, event.y)
        def on_up(event):
            if rect[0]:
                x1, y1, x2, y2 = start[0], start[1], event.x, event.y
                boxes.append((x1, y1, x2, y2))
                canvas.create_rectangle(x1, y1, x2, y2, outline='green')
                rect[0] = None
        canvas.bind('<Button-1>', on_down)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<ButtonRelease-1>', on_up)
        def on_next():
            root.destroy()
        btn = tk.Button(root, text='Дальше', command=on_next)
        btn.pack()
        root.mainloop()
        if boxes:
            with open(label_file, 'w', encoding='utf-8') as f:
                for x1, y1, x2, y2 in boxes:
                    xc = ((x1 + x2) / 2) / tkimg.width()
                    yc = ((y1 + y2) / 2) / tkimg.height()
                    bw = abs(x2 - x1) / tkimg.width()
                    bh = abs(y2 - y1) / tkimg.height()
                    f.write(f'{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')
    print('Разметка завершена.')

def prepare_and_train():
    class_map = {}
    if os.path.exists('class_map.txt'):
        with open('class_map.txt', 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    cid, cname = line.strip().split(':')
                    class_map[cname] = int(cid)
    class_list = [None] * len(class_map)
    for cname, cid in class_map.items():
        class_list[cid] = cname
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(DATASET_DIR, sub), exist_ok=True)
    imgs = [f for f in os.listdir(RAW_DIR) if f.endswith('.png')]
    random.shuffle(imgs)
    split = int(len(imgs) * TRAIN_RATIO)
    train, val = imgs[:split], imgs[split:]
    for subset, files in [('train', train), ('val', val)]:
        for f in files:
            img_src = os.path.join(RAW_DIR, f)
            img_dst = os.path.join(IMAGES_DIR, subset, f)
            shutil.copy(img_src, img_dst)
            label_src = os.path.join(LABELS_DIR, f.replace('.png', '.txt'))
            label_dst = os.path.join(LABELS_YOLO_DIR, subset, f.replace('.png', '.txt'))
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
    with open(CLASSES_TXT, 'w', encoding='utf-8') as f:
        for cname in class_list:
            f.write(cname + '\n')
    with open(DATASET_YAML, 'w', encoding='utf-8') as f:
        f.write(f"train: {os.path.abspath(IMAGES_DIR + '/train')}\n")
        f.write(f"val: {os.path.abspath(IMAGES_DIR + '/val')}\n\n")
        f.write(f"nc: {len(class_list)}\n")
        f.write(f"names: {class_list}\n")
    import subprocess
    cmd = [
        'python', '-m', 'yolov5.train',
        '--img', str(IMG_SIZE),
        '--batch', '16',
        '--epochs', '150',
        '--data', DATASET_YAML,
        '--weights', '',
        '--cfg', YOLOV5_CFG,
        '--device', '0',
        '--name', 'auto_yolo_train'
    ]
    print(' '.join(cmd))
    subprocess.run(cmd)

def yolo_screen_viewer():
    if not os.path.exists(MODEL_PATH):
        print('Сначала обучите модель!')
        return
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    sct = mss.mss()
    monitor = sct.monitors[1]
    root = tk.Tk()
    root.title('YOLOv5: зрение (полный экран)')
    root.attributes('-topmost', True)
    label = tk.Label(root)
    label.pack()
    os.makedirs('detected_screens', exist_ok=True)
    log_path = 'detected_log.txt'
    def update():
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        results = model(img, size=IMG_SIZE)
        draw = ImageDraw.Draw(img)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            if conf >= 0.7:
                x1, y1, x2, y2 = map(int, xyxy)
                label_str = f'{model.names[int(cls)]} {conf:.2f}'
                draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                draw.text((x1, y1), label_str, fill='green')
                detections.append((x1, y1, x2, y2, model.names[int(cls)], conf))
        if detections:
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f'detected_{now}.png'
            img.save(os.path.join('detected_screens', fname))
            with open(log_path, 'a', encoding='utf-8') as logf:
                for x1, y1, x2, y2, cname, conf in detections:
                    logf.write(f'{now} {fname} {cname} {conf:.3f} bbox=({x1},{y1},{x2},{y2})\n')
        img_disp = img.resize((int(monitor['width'] * 0.5), int(monitor['height'] * 0.5)))
        tkimg = ImageTk.PhotoImage(img_disp)
        label.config(image=tkimg)
        label.image = tkimg
        root.after(REFRESH_MS, update)
    update()
    root.mainloop()

def main_menu():
    while True:
        print('\n--- YOLOv5 Pipeline ---')
        print('1. Собирать скриншоты')
        print('2. Размечать скриншоты')
        print('3. Обучить модель')
        print('4. Запустить "зрение" нейронки')
        print('5. Выйти')
        choice = input('Выберите действие: ')
        if choice == '1':
            collect_screenshots()
        elif choice == '2':
            label_screenshots()
        elif choice == '3':
            prepare_and_train()
        elif choice == '4':
            yolo_screen_viewer()
        elif choice == '5':
            break
        else:
            print('Некорректный выбор!')

if __name__ == '__main__':
    main_menu() 