import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 모델과 데이터셋 설정
model_name = "distilbert-base-uncased"
dataset = load_dataset("imdb")

# 데이터 전처리
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train_datasets = dataset["train"].map(tokenize_function, batched=True)
tokenized_test_datasets = dataset["test"].map(tokenize_function, batched=True)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",           # 결과 저장 경로
    num_train_epochs=1,               # 훈련 에폭 수 1로 설정 (빠르게 테스트)
    per_device_train_batch_size=8,    # 배치 크기 줄이기 (CPU에서는 작은 값 추천)
    per_device_eval_batch_size=8,     # 배치 크기 줄이기
    evaluation_strategy="epoch",      # 에폭마다 검증
    save_strategy="epoch",            # 체크포인트 저장 전략 (에폭마다 저장)
    logging_dir="./logs",             # 로그 저장 경로
    logging_steps=100,                # 100 스텝마다 로그 출력
    load_best_model_at_end=True,      # 최상의 모델로 종료
    use_cpu=True                      # CPU 사용 설정
)

# Trainer 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_test_datasets,
)

trainer.train()
