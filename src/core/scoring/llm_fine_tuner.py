#!/usr/bin/env python3
"""
LLM Fine-Tuning Pipeline for Milestone Scoring

This module implements a fine-tuning pipeline for Mistral models to improve 
developmental milestone assessment scoring.
"""

import os
import sys
import json
import logging
import random
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_fine_tuner")

@dataclass
class FineTuningExample:
    """Represents a single fine-tuning example for milestone scoring."""
    milestone: str
    domain: str
    age_range: str
    response: str
    expected_score: str
    reasoning: Optional[str] = None

class LLMFineTuner:
    """
    Fine-tuning pipeline for milestone scoring models.
    
    This class handles data preparation, model fine-tuning, and model management
    for improving milestone scoring capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM fine-tuning pipeline.
        
        Args:
            config: Configuration dictionary for fine-tuning parameters
        """
        self.config = config or self._default_config()
        self.data_dir = Path(self.config.get("data_dir", "data/fine_tuning"))
        self.models_dir = Path(self.config.get("models_dir", "models/fine-tuned"))
        
        # Create necessary directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load LoRA library if available
        try:
            import peft
            self.peft_available = True
        except ImportError:
            logger.warning("PEFT library not available. Fine-tuning will be limited.")
            self.peft_available = False
            
        # Initialize tracking
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.models_dir / f"run_{self.current_run_id}"
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": []
        }
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration for fine-tuning."""
        return {
            "base_model": "models/mistral-7b-instruct-v0.2.Q3_K_S.gguf",
            "output_dir": "models/fine-tuned/milestone-scorer",
            "data_dir": "data/fine_tuning",
            "epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 1,
            "fp16": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "train_test_split": 0.2,
            "max_steps": 1000,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 200,
            "prompt_template": """
<s>[INST] You are an expert in child development trained to score developmental milestones. 
 
Milestone: {milestone}
Domain: {domain}
Age Range: {age_range}
Parent's Response: {response}

Rate the child's ability on this milestone using one of the following scores:
0 - CANNOT_DO: Child cannot perform the skill at all
1 - EMERGING: Child shows beginning signs of the skill
2 - WITH_SUPPORT: Child can do with help or prompting
3 - INDEPENDENT: Child can do consistently without help

Provide your score and detailed reasoning for your decision.
[/INST]
"""
        }
    
    def prepare_fine_tuning_data(self, 
                               gold_standard_data: List[Dict[str, Any]], 
                               format: str = "mistral-instruct") -> List[Dict[str, str]]:
        """
        Prepare training data from gold standard dataset.
        
        Args:
            gold_standard_data: List of gold standard examples with expert scoring
            format: Format of training data (e.g., "mistral-instruct")
            
        Returns:
            List of formatted training examples
        """
        training_data = []
        
        # Process each example in the gold standard dataset
        for example in gold_standard_data:
            milestone = example.get("milestone", "")
            domain = example.get("domain", "")
            age_range = example.get("age_range", "")
            response = example.get("response", "")
            score = example.get("score", "")
            reasoning = example.get("reasoning", "")
            
            # Format according to model requirements
            if format == "mistral-instruct":
                prompt = self.config["prompt_template"].format(
                    milestone=milestone,
                    domain=domain,
                    age_range=age_range,
                    response=response
                )
                
                # Construct completion based on the score and reasoning
                completion = f"""Based on the parent's response, I rate this as {score}.

Reasoning: {reasoning}

Final Score: {score}"""
                
                training_data.append({
                    "prompt": prompt.strip(),
                    "completion": completion.strip()
                })
                
        # Save the prepared data
        self._save_training_data(training_data)
        logger.info(f"Prepared {len(training_data)} examples for fine-tuning")
        
        return training_data
    
    def _save_training_data(self, data: List[Dict[str, str]]) -> None:
        """Save training data to disk."""
        output_file = self.data_dir / f"training_data_{self.current_run_id}.jsonl"
        
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved training data to {output_file}")
    
    def split_training_data(self, 
                          training_data: List[Dict[str, str]], 
                          validation_ratio: float = 0.2) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split training data into training and validation sets.
        
        Args:
            training_data: List of formatted training examples
            validation_ratio: Ratio of data to use for validation
            
        Returns:
            Tuple of (train_data, val_data)
        """
        # Shuffle data for random split
        shuffled_data = training_data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split point
        split_idx = int(len(shuffled_data) * (1 - validation_ratio))
        
        # Split the data
        train_data = shuffled_data[:split_idx]
        val_data = shuffled_data[split_idx:]
        
        # Save the splits
        train_file = self.data_dir / f"train_{self.current_run_id}.jsonl"
        val_file = self.data_dir / f"val_{self.current_run_id}.jsonl"
        
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
                
        with open(val_file, 'w') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")
        
        return train_data, val_data
    
    def fine_tune_model(self, 
                       train_data: List[Dict[str, str]], 
                       val_data: List[Dict[str, str]], 
                       config: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute fine-tuning on the model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            config: Fine-tuning configuration
            
        Returns:
            Path to the fine-tuned model
        """
        # Use provided config or default
        ft_config = config or self.config
        
        # Create model output directory
        output_dir = Path(ft_config.get("output_dir", self.run_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to LoRA fine-tuning if PEFT is available
        if self.peft_available:
            logger.info("Using PEFT for LoRA fine-tuning")
            return self._lora_fine_tune(train_data, val_data, ft_config)
        else:
            logger.info("PEFT not available, using basic fine-tuning")
            return self._basic_fine_tune(train_data, val_data, ft_config)
    
    def _lora_fine_tune(self, 
                       train_data: List[Dict[str, str]], 
                       val_data: List[Dict[str, str]], 
                       config: Dict[str, Any]) -> str:
        """
        Execute LoRA fine-tuning with the PEFT library.
        
        Args:
            train_data: Training data
            val_data: Validation data
            config: Fine-tuning configuration
            
        Returns:
            Path to the fine-tuned model
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            import torch
            from datasets import Dataset
            
            # Load base model and tokenizer
            base_model = config.get("base_model")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Prepare model for LoRA training
            model = prepare_model_for_kbit_training(model)
            
            # Set up LoRA configuration
            lora_config = LoraConfig(
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA config to model
            model = get_peft_model(model, lora_config)
            
            # Prepare datasets
            def tokenize_function(examples):
                # Combine prompt and completion for training
                texts = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
                return tokenizer(texts, padding="max_length", truncation=True, max_length=512)
            
            # Create datasets
            train_dataset = Dataset.from_list([{"prompt": x["prompt"], "completion": x["completion"]} for x in train_data])
            val_dataset = Dataset.from_list([{"prompt": x["prompt"], "completion": x["completion"]} for x in val_data])
            
            # Tokenize
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            val_dataset = val_dataset.map(tokenize_function, batched=True)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=config.get("output_dir"),
                num_train_epochs=config.get("epochs", 3),
                per_device_train_batch_size=config.get("batch_size", 1),
                per_device_eval_batch_size=config.get("batch_size", 1),
                learning_rate=config.get("learning_rate", 2e-5),
                fp16=config.get("fp16", True),
                logging_dir=f"{config.get('output_dir')}/logs",
                logging_steps=config.get("logging_steps", 10),
                save_steps=config.get("save_steps", 200),
                max_steps=config.get("max_steps", 1000),
                warmup_steps=config.get("warmup_steps", 100),
                evaluation_strategy="steps",
                eval_steps=100,
                save_total_limit=3,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            # Start training
            logger.info("Starting LoRA fine-tuning")
            trainer.train()
            
            # Save the fine-tuned model
            model.save_pretrained(f"{config.get('output_dir')}/final")
            tokenizer.save_pretrained(f"{config.get('output_dir')}/final")
            
            # Save training metadata
            with open(f"{config.get('output_dir')}/training_metadata.json", 'w') as f:
                json.dump({
                    "base_model": base_model,
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "config": config,
                    "timestamp": self.current_run_id
                }, f, indent=2)
            
            logger.info(f"Fine-tuning completed. Model saved to {config.get('output_dir')}/final")
            return f"{config.get('output_dir')}/final"
            
        except Exception as e:
            logger.error(f"Error during LoRA fine-tuning: {str(e)}")
            return ""
    
    def _basic_fine_tune(self, 
                        train_data: List[Dict[str, str]], 
                        val_data: List[Dict[str, str]], 
                        config: Dict[str, Any]) -> str:
        """
        Execute basic fine-tuning without PEFT.
        
        Args:
            train_data: Training data
            val_data: Validation data
            config: Fine-tuning configuration
            
        Returns:
            Path to the fine-tuned model
        """
        logger.warning("Basic fine-tuning method not implemented - requires high GPU memory")
        logger.warning("Consider installing PEFT for more efficient fine-tuning with LoRA")
        
        # Save the configuration for future reference
        output_dir = config.get("output_dir")
        with open(f"{output_dir}/attempted_config.json", 'w') as f:
            json.dump({
                "train_size": len(train_data),
                "val_size": len(val_data),
                "config": config,
                "timestamp": self.current_run_id,
                "status": "failed - PEFT required"
            }, f, indent=2)
        
        return ""
    
    def load_gold_standard(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load gold standard dataset for fine-tuning.
        
        Args:
            file_path: Path to gold standard dataset (JSON or CSV)
            
        Returns:
            List of gold standard examples
        """
        if file_path is None:
            file_path = os.environ.get("GOLD_STANDARD_PATH", "data/gold_standard.json")
        
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Gold standard file not found: {file_path}")
            return []
        
        # Load based on file type
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                # Handle case where data is a dict with examples as a key
                data = data.get("examples", [])
                
            return data
        
        elif path.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(path)
                return df.to_dict('records')
            except Exception as e:
                logger.error(f"Error loading CSV: {str(e)}")
                return []
        
        else:
            logger.error(f"Unsupported file format: {path.suffix}")
            return []
    
    def evaluate_model(self, model_path: str, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate a fine-tuned model on test data.
        
        Args:
            model_path: Path to the fine-tuned model
            test_data: Test examples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            correct = 0
            total = 0
            
            for example in test_data:
                # Format prompt
                prompt = self.config["prompt_template"].format(
                    milestone=example.get("milestone", ""),
                    domain=example.get("domain", ""),
                    age_range=example.get("age_range", ""),
                    response=example.get("response", "")
                )
                
                # Generate response
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=200,
                        temperature=0.1
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract score from response
                expected_score = example.get("score", "")
                if expected_score in response:
                    correct += 1
                total += 1
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            
            # Return evaluation metrics
            metrics = {
                "accuracy": accuracy,
                "total_examples": total,
                "model_path": model_path
            }
            
            # Log and save metrics
            logger.info(f"Model evaluation completed with accuracy: {accuracy:.2f}")
            with open(f"{self.run_dir}/evaluation_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return {"accuracy": 0, "error": str(e)}

def fine_tune_llm_for_milestone_scoring(config_override: Optional[Dict[str, Any]] = None):
    """
    Set up fine-tuning for the Mistral model for milestone scoring.
    
    Args:
        config_override: Optional configuration overrides
    """
    # Create fine-tuning manager
    fine_tuner = LLMFineTuner(config_override)
    
    # Prepare training data from gold standard dataset
    gold_standard = fine_tuner.load_gold_standard()
    if not gold_standard:
        logger.error("No gold standard data available for fine-tuning")
        return
    
    # Prepare and format the data
    training_data = fine_tuner.prepare_fine_tuning_data(
        gold_standard, 
        format="mistral-instruct"
    )
    
    # Split data for training/validation
    train_data, val_data = fine_tuner.split_training_data(
        training_data, 
        validation_ratio=0.2
    )
    
    # Define fine-tuning configuration (or use the default if none provided)
    ft_config = config_override or fine_tuner.config
    
    # Execute fine-tuning
    model_path = fine_tuner.fine_tune_model(train_data, val_data, ft_config)
    
    # Evaluate if fine-tuning was successful
    if model_path:
        # Use a portion of validation data for testing
        test_data = val_data[:min(len(val_data), 50)]
        metrics = fine_tuner.evaluate_model(model_path, test_data)
        
        logger.info(f"Fine-tuning completed successfully. Model saved to {model_path}")
        logger.info(f"Evaluation metrics: {metrics}")
    else:
        logger.error("Fine-tuning failed. See logs for details.")

if __name__ == "__main__":
    # Example usage when run as a script
    fine_tune_llm_for_milestone_scoring() 
 