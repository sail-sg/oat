from datasets import load_dataset
import zipfile
import urllib.request

github_url = "https://github.com/hendrycks/math/archive/refs/heads/main.zip"

urllib.request.urlretrieve(github_url, "math_dataset.zip")

with zipfile.ZipFile("math_dataset.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

def prepare_data():
    print("Downloading the MATH dataset from Hugging Face...")
    dataset = load_dataset("hendrycks/competition_math", split='train')
    split_dataset = dataset.train_test_split(test_size=0.02, seed=42)
    train_data = split_dataset['train']
    test_data = split_dataset['test']
    print("\n--- Displaying 10 sample problems for manual curation ---")
    print("You can use these to create your few-shot and test examples.\n")
    shuffled_dataset = dataset.shuffle(seed=42)
    for i in range(10):
        item = shuffled_dataset[i]
        print(f"--- Example {i+1} ---")
        print(f"Level: {item['level']}")
        print(f"Type: {item['type']}")
        print(f"Question: {item['problem']}")
        print(f"Correct Solution: {item['solution']}\n")

if __name__ == "__main__":
    prepare_data()