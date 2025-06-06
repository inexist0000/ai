from transformers import AutoTokenizer

# 원래 사용한 모델의 기본 토크나이저를 가져옴
model_name = "distilbert-base-uncased"  # 너가 사용한 모델에 맞게 변경

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 기존 checkpoint-3125 폴더에 토크나이저 저장
tokenizer.save_pretrained("./learn/results/checkpoint-3125/")

print("✅ 토크나이저 저장 완료!")
