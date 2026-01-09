"""
测试训练好的模型是否正常工作
运行: python test_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 配置
MODEL_PATH = "./models/modernBERT-multilingual-finetune"
MAX_LENGTH = 256

# 标签映射
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

print("=" * 80)
print("正在加载模型...")
print("=" * 80)

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"✓ 模型加载成功")
print(f"✓ 使用设备: {device}")
print(f"✓ 标签映射: {id2label}")
print()


def predict_sentiment(text):
    """预测单个文本的情感"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    pred_label = id2label[pred_id]
    all_probs = probs[0].cpu().numpy()

    return pred_label, confidence, all_probs


# 测试样本
test_texts = [
    "I love this product! It's amazing and works perfectly!",
    "This is okay, nothing special about it.",
    "I hate this service. It's terrible and completely useless.",
    "这个产品非常好用，强烈推荐！",
    "这个电影一般般，没什么特别的。",
    "太差了，完全不推荐！",
    "このビデオは嫌いです。",
    "Le soleil brille si fort aujourd'hui, je suis si heureuse.",
    "Я хочу пожаловаться на это видео и добиться его удаления."
]

print("=" * 80)
print("测试预测结果")
print("=" * 80)

for i, text in enumerate(test_texts, 1):
    pred_label, confidence, all_probs = predict_sentiment(text)

    print(f"\n文本 #{i}: {text}")
    print(f"  预测: {pred_label} (置信度: {confidence:.4f})")
    print(f"  概率分布:")
    for label_id, label_name in id2label.items():
        print(f"    {label_name}: {all_probs[label_id]:.4f}")
    print("-" * 80)

print("\n✓ 模型测试完成！可以正常使用。")