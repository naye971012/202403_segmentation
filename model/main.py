from transformers import AutoImageProcessor
from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer, SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

import argparse
import torch
import numpy as np
import wandb
import warnings
warnings.filterwarnings("ignore")

from trainer import *
from dataset import get_dataset
from utils import compute_metrics, set_seed

id2label = {
    0: "fire",
    #1: "fire"
}
label2id = {
    "fire": 0,
    #"fire":1
}

def get_args():
    parser = argparse.ArgumentParser(description="Arguments for training")

    # Required arguments
    parser.add_argument("--output_dir", type=str, default="./Segformer_FocalDice-ratio1",
                        help="Directory where the model checkpoints and outputs will be saved")
    parser.add_argument("--model_name", type=str, default="nvidia/MiT-b5",
                        help="baseline model name")    

    parser.add_argument("--data_path", type=str, default="../data",
                        help="data folder path")    
    
    parser.add_argument("--validation_ratio", type=float, default=0.1,
                        help="train test split ratio")    
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for reproduce")    

    args = parser.parse_args()
    return args


def main():
    set_seed()
    
    args = get_args()
    
    train_dataset, valid_dataset, eval_dataset = get_dataset(args)    

    out = train_dataset[0]
    print(f"image shape: {out['image'].shape}")
    print(f"mask  shape: {out['annotation'].shape}")

    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name, 
                                                             id2label=id2label, 
                                                             label2id=label2id,
                                                             ignore_mismatched_sizes=True
                                                             )
    
    global image_processor
    image_processor = SegformerImageProcessor.from_pretrained(args.model_name,
                                                              do_reduce_labels=False) #True 

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=1e-4,
        #warmup_steps=1000,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        eval_steps=500,
        logging_steps=30,
        gradient_accumulation_steps=2,
        eval_accumulation_steps=4,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
    )

    wandb.init(project="2024-SPARK-6",
               name="Baseline-Dice+Focal-Ratio-1")

    trainer = FocalDiceTrainer(
        data_collator=collate_fn,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


#datalodaer collate_fn $더 깔끔하게 짤 수 있을텐데...
def collate_fn(example_batch):

        have_label = False
        if 'annotation' in example_batch[0].keys():
            have_label = True
        
        images = []
        labels = []
        #for train/validation
        if have_label:
            for x in example_batch:
                images.append(x["image"])
                labels.append(x["annotation"])  # (batch_size, H, W)

            inputs = image_processor.preprocess(images, 
                                                labels, 
                                                do_normalize=True,
                                                do_rescale=False,
                                                do_resize=False,
                                                do_reduce_labels=False, 
                                                return_tensors="pt")
        #for testing process
        else:
            for x in example_batch:
                images.append(x["image"])
            inputs = image_processor.preprocess(images,
                                                do_normalize=True,
                                                do_rescale=False,
                                                do_resize=False,
                                                do_reduce_labels=False, 
                                                return_tensors="pt")
        return inputs


if __name__=="__main__":
    main()
    
