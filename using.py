from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./learn/results/checkpoint-3125/"  # 저장된 모델 경로

# 토크나이저와 모델 로드
# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 테스트 입력값
text = "This movie was too bad!"
inputs = tokenizer(text, return_tensors="pt")

# 모델 예측
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax().item()

print(f"예측된 클래스: {predicted_class}")  # 1(긍정) 또는 0(부정)
