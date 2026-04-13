from datasets import load_dataset
import random


def _load_bigbio_subset_parquet(dataset_repo, subset_dir, split='train'):
    supported_splits = {"train", "validation", "test"}
    if split not in supported_splits:
        raise ValueError(f"Unsupported split '{split}'. Use one of: {sorted(supported_splits)}")

    parquet_url = (
        f"hf://datasets/{dataset_repo}@refs/convert/parquet/"
        f"{subset_dir}/{split}/0000.parquet"
    )
    return load_dataset("parquet", data_files={split: parquet_url}, split=split)

# MedQA dataset loader
def load_medqa_dataset(split='train'):
    dataset = _load_bigbio_subset_parquet(
        dataset_repo='bigbio/med_qa',
        subset_dir='med_qa_en_4options_source',
        split=split,
    )
    return dataset

# Sample a subset of the dataset for quick testing
def sample_dataset(dataset, num_samples=100):
    if num_samples > len(dataset):
        num_samples = len(dataset)
    sampled_indices = random.sample(range(len(dataset)), num_samples)
    sampled_dataset = dataset.select(sampled_indices)
    return sampled_dataset

# PubMedQA dataset loader
def load_pubmedqa_dataset(split='train'):
    dataset = _load_bigbio_subset_parquet(
        dataset_repo='bigbio/pubmed_qa',
        subset_dir='pubmed_qa_labeled_fold0_source',
        split=split,
    )
    return dataset

if __name__ == "__main__":
    # Load the MedQA dataset
    medqa_dataset = load_medqa_dataset()
    
    # Sample a subset for testing
    sampled_medqa = sample_dataset(medqa_dataset, num_samples=100)
    
    # Print a few samples to verify
    print("Sampled MedQA Dataset:")
    for i in range(5):
        print(sampled_medqa[i])
    
    # Load the PubMedQA dataset
    pubmedqa_dataset = load_pubmedqa_dataset()

    # Sample a subset for testing
    sampled_pubmedqa = sample_dataset(pubmedqa_dataset, num_samples=100)
    print("Sampled PubMedQA Dataset:")
    for i in range(5):
        print(sampled_pubmedqa[i])