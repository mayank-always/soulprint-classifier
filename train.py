from setfit import SetFitModel, SetFitTrainer, SetFitTrainingArguments
from datasets import Dataset
from dataset import train_texts, train_labels

# Prepare dataset
dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels
}).train_test_split(test_size=0.1)

# Load model
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2",
    use_differentiable_head=False
)

# Train setup
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    loss_class="CosineSimilarityLoss",
    batch_size=16,
    num_iterations=20,
    num_epochs=5
)

# Train
trainer.train()

# Test
sample_inputs = [
    "I love preserving memories and family stories.",
    "I'm always organizing others and standing for what’s right.",
    "It brings me joy to make people smile through tough times."
]

preds = trainer.model.predict(sample_inputs)

for i, p in zip(sample_inputs, preds):
    print(f">> {i} → {p}")

# Optional save
model.save_pretrained("soulprint_setfit_model")
