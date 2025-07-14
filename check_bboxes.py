import os
from PIL import Image, ImageDraw

RAW_DIR = 'raw_screens'
LABELS_DIR = 'labels'
OUT_DIR = 'check_bboxes_vis'
os.makedirs(OUT_DIR, exist_ok=True)

for img_file in os.listdir(RAW_DIR):
    if not img_file.endswith('.png'):
        continue
    img_path = os.path.join(RAW_DIR, img_file)
    label_path = os.path.join(LABELS_DIR, img_file.replace('.png', '.txt'))
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, bw, bh = parts
                    xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    draw.text((x1, y1), f'{cls}', fill='red')
    img.save(os.path.join(OUT_DIR, img_file))
print(f'Проверь папку {OUT_DIR} — там все скрины с нарисованными bbox которые ты сделал.') 