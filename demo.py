"""
demo.py
-------
CIFAR-100 互動式展示 Demo（Gradio）

Colab 執行：
    !pip install -q gradio
    !python demo.py --checkpoint /content/drive/MyDrive/cifar100_checkpoints/model_best.pth --share

本機執行：
    pip install gradio
    python demo.py --checkpoint ./checkpoints/model_best.pth
"""

import argparse
import random
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image
import gradio as gr

from model import get_model


# CIFAR-100 類別名稱
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm',
]

# 全域變數
MODEL         = None
DEVICE        = None
SAMPLE_IMAGES = []  # [(pil_image, true_label_idx), ...]

# 預處理（與訓練時一致）
TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


def load_model(checkpoint_path, num_classes, device):
    model = get_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    best_acc = checkpoint.get("best_acc", "N/A")
    print(f"✅ Model loaded | best_acc={best_acc:.2f}%" if isinstance(best_acc, float) else f"✅ Model loaded")
    return model


def load_sample_images(data_dir, n=30):
    """從 CIFAR-100 測試集隨機抽取樣本圖片"""
    dataset = datasets.CIFAR100(root=data_dir, train=False, download=True)
    indices = random.sample(range(len(dataset)), n)
    samples = []
    for idx in indices:
        img, label = dataset[idx]
        # 放大顯示（32x32 太小）
        img_large = img.resize((128, 128), Image.NEAREST)
        samples.append((img_large, label))
    return samples


def predict(image):
    """輸入 PIL Image，回傳 Top-5 預測結果"""
    if image is None:
        return {}

    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    top5_probs, top5_idx = torch.topk(probs, 5)
    return {
        CIFAR100_CLASSES[idx.item()]: round(prob.item(), 4)
        for prob, idx in zip(top5_probs, top5_idx)
    }


def get_random_sample():
    """隨機抽一張樣本圖片"""
    if not SAMPLE_IMAGES:
        return None, "No samples loaded"
    img, label = random.choice(SAMPLE_IMAGES)
    return img, f"✅ True Label: {CIFAR100_CLASSES[label]}"


def build_demo():
    with gr.Blocks(title="CIFAR-100 Classifier | CIFARMLOps") as demo:

        gr.Markdown("""
        # 🖼️ CIFAR-100 Image Classifier
        **Model:** ResNet-18 trained on CIFAR-100 (100 classes)
        **Project:** [CIFARMLOps](https://github.com/rvgoing/CIFARMLOps) — A progressively evolving MLOps system

        Upload any image **or** click **🎲 Random Sample** to try a test image from the CIFAR-100 dataset.
        The model returns the **Top-5 predicted classes** with confidence scores.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image    = gr.Image(type="pil", label="📤 Input Image")
                true_label_box = gr.Textbox(label="Ground Truth（Random Sample 才有）", interactive=False)
                with gr.Row():
                    btn_predict = gr.Button("🔍 Predict", variant="primary")
                    btn_sample  = gr.Button("🎲 Random Sample", variant="secondary")

            with gr.Column(scale=1):
                sample_display = gr.Image(label="🖼️ Sample Preview", interactive=False)
                output_label   = gr.Label(num_top_classes=5, label="📊 Top-5 Predictions")

        # Random Sample：顯示樣本圖 + true label + 自動預測
        def on_random_sample():
            img, true_label = get_random_sample()
            result = predict(img) if img else {}
            return img, img, true_label, result

        btn_sample.click(
            fn=on_random_sample,
            outputs=[input_image, sample_display, true_label_box, output_label]
        )

        # Predict：對目前輸入圖片預測
        btn_predict.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_label]
        )

        gr.Markdown("""
        ---
        > **⚠️ Note:** CIFAR-100 images are natively **32×32 pixels**.
        > Uploaded high-resolution images will be resized to 32×32 before inference.
        > Accuracy on real-world photos may vary — this model is optimized for CIFAR-100 style images.
        """)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 Gradio Demo")
    parser.add_argument("--checkpoint",  required=True, type=str, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data-dir",    default="./data", type=str, help="Path to CIFAR-100 dataset")
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--num-samples", default=30, type=int, help="Number of random samples to preload")
    parser.add_argument("--share",       action="store_true", help="Generate public Gradio URL (for Colab)")
    parser.add_argument("--port",        default=7860, type=int)
    return parser.parse_args()


def main():
    global MODEL, DEVICE, SAMPLE_IMAGES

    args   = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    MODEL         = load_model(args.checkpoint, args.num_classes, DEVICE)
    SAMPLE_IMAGES = load_sample_images(args.data_dir, n=args.num_samples)
    print(f"✅ Loaded {len(SAMPLE_IMAGES)} sample images")

    demo = build_demo()
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()