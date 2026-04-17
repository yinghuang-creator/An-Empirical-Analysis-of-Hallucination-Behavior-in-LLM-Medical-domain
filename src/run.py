import os
import torch

from dataset_loader import load_medqa_dataset, sample_dataset

from baseline_eval import load_model, evaluate_combined_batch

def main():
    # Configurations
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    NUM_SAMPLES = 200
    BATCH_SIZE = 8
    
    # Set the output path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'medqa_combined_results.csv')

    # Load and sample dataset
    print(f"Loading MedQA dataset and sampling {NUM_SAMPLES} examples...")
    medqa_dataset = load_medqa_dataset() 
    sampled_data = sample_dataset(medqa_dataset, num_samples=NUM_SAMPLES)
    
    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}...")
    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"):
        print("Using Hugging Face token from environment.")
    tokenizer, model = load_model(MODEL_NAME)
    
    # Start combined evaluation
    print(f"Starting Combined Evaluation (Baseline vs CoT)...")
    print(f"Results will be saved to: {output_path}")
    
    results = evaluate_combined_batch(
        dataset=sampled_data, 
        tokenizer=tokenizer, 
        model=model, 
        batch_size=BATCH_SIZE,
        output_csv=output_path
    )
    
    print("\nEvaluation Complete!")
    print(f"Total Questions  : {results['total']}")
    print(f"Baseline Correct : {results['baseline_correct']}")
    print(f"Baseline Accuracy: {results['baseline_correct'] / results['total']:.2%}")
    print(f"CoT Correct      : {results['cot_correct']}")
    print(f"CoT Accuracy     : {results['cot_correct'] / results['total']:.2%}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()