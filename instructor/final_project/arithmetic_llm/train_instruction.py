"""Instruction fine-tuning script for arithmetic LLM.

This module implements the fine-tuning pipeline for the instruction-tuned model
on instruction-formatted arithmetic data.
"""

import os
import json
import torch
from typing import Optional, Dict, Any, List

from .transformer_model import ArithmeticTransformer
from .arithmetic_tokenizer import ArithmeticBPETokenizer
from .data_loader import create_dataloaders
from .training_config import TrainingConfig
from .output_naming import create_numbered_output_dir
from .train_foundational import (
    get_linear_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    train_epoch,
    evaluate
)


def train_instruction_model(
    instruction_corpus_path: str,
    tokenizer_path: str,
    foundational_checkpoint: str,
    output_dir: str,
    config: TrainingConfig,
    model_config: Optional[Dict] = None,
    wandb_enabled: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    wandb_mode: str = "online",
) -> str:
    """Fine-tune model with instruction formatting.
    
    Args:
        instruction_corpus_path: Path to instruction-formatted corpus
        tokenizer_path: Path to trained tokenizer
        foundational_checkpoint: Path to foundational model checkpoint
        output_dir: Directory to save checkpoints and logs
        config: Training configuration
        model_config: Optional model architecture configuration
        
    Returns:
        Path to final fine-tuned model checkpoint
    """
    # Validate configuration
    config.validate()
    
    # Create unique output directory with numeric run index
    output_dir = create_numbered_output_dir(output_dir, "instruction")
    
    print(f"Fine-tuning output directory: {output_dir}")
    print(f"Configuration: {config.to_dict()}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2id)
    print(f"Tokenizer vocabulary size: {vocab_size}")
    
    # Initialize model architecture
    print("Initializing model architecture...")
    checkpoint_data = torch.load(foundational_checkpoint, map_location='cpu')
    if model_config is None:
        model_config = checkpoint_data.get('model_config', {
            'vocab_size': vocab_size,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'max_seq_length': 512
        })
    else:
        model_config['vocab_size'] = vocab_size

    # Validate tokenizer vocab size against checkpoint if present
    checkpoint_vocab_size = checkpoint_data.get('tokenizer_vocab_size')
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({vocab_size}) does not match checkpoint "
            f"vocab size ({checkpoint_vocab_size})."
        )

    model_config['vocab_size'] = vocab_size
    max_seq_length = model_config.get('max_seq_length', 512)

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        corpus_path=instruction_corpus_path,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=max_seq_length,
        train_split=0.9,
        shuffle=True,
        num_workers=0,
        mode="instruction",
        split_strategy="group_by_expression",
        split_seed=42,
    )
    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    model = ArithmeticTransformer(**model_config)
    
    # Load foundational model weights
    print(f"Loading foundational model from: {foundational_checkpoint}")
    checkpoint_metadata = load_checkpoint(
        checkpoint_path=foundational_checkpoint,
        model=model
    )
    print(f"Loaded checkpoint from epoch {checkpoint_metadata['epoch']}, "
          f"step {checkpoint_metadata['step']}")
    
    model = model.to(config.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config.num_epochs
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    wandb_run: Optional[Any] = None
    if wandb_enabled:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging requested, but wandb is not installed. "
                "Install with: pip install wandb"
            ) from exc

        run_tags = ["instruction"]
        if wandb_tags:
            run_tags.extend(wandb_tags)

        wandb_run = wandb.init(
            project=wandb_project or "stat359-arithmetic-llm",
            entity=wandb_entity,
            name=wandb_run_name,
            tags=run_tags,
            mode=wandb_mode,
            config={
                "phase": "instruction",
                "instruction_corpus_path": instruction_corpus_path,
                "tokenizer_path": tokenizer_path,
                "foundational_checkpoint": foundational_checkpoint,
                "output_dir": output_dir,
                "training_config": config.to_dict(),
                "model_config": model_config,
                "total_steps": total_steps,
            },
        )
    
    # Save configuration
    config.to_json(os.path.join(output_dir, 'training_config.json'))
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Save foundational checkpoint path for reference
    with open(os.path.join(output_dir, 'foundational_checkpoint.txt'), 'w') as f:
        f.write(foundational_checkpoint)
    
    # Training loop
    print("\nStarting instruction fine-tuning...")
    global_step = 0
    best_val_loss = float('inf')
    training_log = []
    training_step_log = []

    try:
        for epoch in range(config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"{'='*60}")

            # Train for one epoch
            train_loss, global_step = train_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch + 1,
                global_step=global_step,
                output_dir=output_dir,
                tokenizer_vocab_size=vocab_size,
                wandb_run=wandb_run,
                step_log=training_step_log,
            )

            # Evaluate on validation set
            print("\nEvaluating on validation set...")
            val_loss = evaluate(model, val_dataloader, config)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")

            # Log metrics
            training_log.append({
                'epoch': epoch + 1,
                'step': global_step,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch/train_loss": train_loss,
                        "epoch/val_loss": val_loss,
                        "epoch/learning_rate": scheduler.get_last_lr()[0],
                        "epoch/best_val_loss_so_far": min(best_val_loss, val_loss),
                    },
                    step=global_step,
                )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    step=global_step,
                    loss=val_loss,
                    config=config,
                    tokenizer_vocab_size=vocab_size,
                    output_dir=output_dir,
                    is_final=False
                )
                # Promote to best_model.pt (overwrite if it already exists)
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                os.replace(best_checkpoint_path, best_model_path)
                print(f"  New best model saved: {best_model_path}")

        # Save final model
        print("\nSaving final fine-tuned model...")
        final_checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config.num_epochs,
            step=global_step,
            loss=train_loss,
            config=config,
            tokenizer_vocab_size=vocab_size,
            output_dir=output_dir,
            is_final=True
        )
        print(f"Final model saved: {final_checkpoint_path}")

        # Save training log
        log_path = os.path.join(output_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"Training log saved: {log_path}")

        step_log_path = os.path.join(output_dir, 'training_step_log.json')
        with open(step_log_path, 'w') as f:
            json.dump(training_step_log, f, indent=2)
        print(f"Training step log saved: {step_log_path}")

        # Save summary
        summary = {
            'total_epochs': config.num_epochs,
            'total_steps': global_step,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'model_config': model_config,
            'training_config': config.to_dict(),
            'tokenizer_vocab_size': vocab_size,
            'foundational_checkpoint': foundational_checkpoint,
            'split_strategy': 'group_by_expression'
        }
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved: {summary_path}")

        if wandb_run is not None:
            wandb_run.summary["final_train_loss"] = train_loss
            wandb_run.summary["final_val_loss"] = val_loss
            wandb_run.summary["best_val_loss"] = best_val_loss
            wandb_run.summary["total_steps"] = global_step

        print("\n" + "="*60)
        print("Instruction fine-tuning completed successfully!")
        print(f"Output directory: {output_dir}")
        print("="*60)

        return final_checkpoint_path
    finally:
        if wandb_run is not None:
            wandb_run.finish()


