import gc
from tqdm import tqdm
from torch.utils.data import DataLoader

eval_dataloader = DataLoader(
    common_voice_test, batch_size=5, collate_fn=data_collator)
model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    if step > 20:
        break

    with torch.no_grad():
        logits = model(batch["input_features"].to("cuda")).logits

    labels = batch["labels"].cpu().numpy()
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

    decoded_preds = processor.batch_decode(pred_ids)
    decoded_labels = processor.batch_decode(labels)
    evaluation_metric.add_batch(predictions=decoded_preds,
                                references=decoded_labels)
    del logits, labels, batch, pred_ids
    gc.collect()

wer = 100 * evaluation_metric.compute()
print(f"{wer=}")
