# Qwen_SFT
Finetuning experiments with Qwen3

## Example Training Configuration

Here's an example training configuration using the trl library's SFTTrainer and SFTConfig:

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    dataset_text_field="messages",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,                # Increased for more stable start
    num_train_epochs=10,
    learning_rate= 2e-6,             #1e-5             # Lowered for better generalization
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="none",
    eval_strategy="epoch",          # Evaluate and save every epoch
    save_strategy="epoch",
    save_total_limit=2,             # Only keep last 3 checkpoints
    max_grad_norm=1.0,              # Clip gradients
    fp16=True,                      # Enable mixed-precision if available (optional, can remove if not supported)
    push_to_hub=False,  
    neftune_noise_alpha=5,
    assistant_only_loss=True,
    chat_template_path="/content/qwen_chat_template.jinja"
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```
