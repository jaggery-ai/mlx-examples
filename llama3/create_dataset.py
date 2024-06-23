import datasets, os, json
from tqdm import tqdm


# Dataset upload
# from datasets import load_dataset
# ds = load_dataset(os.path.abspath('../../jaggery_dataset'), trust_remote_code=True)
# ds.push_to_hub('jaggery-ai/conversations', private=True)

def to_dialog(example: str, system_prompt: str):
        """Converts the example to dialog format."""
        example = example.split('\n')
        dialog = list()
        dialog.append({"role": "system", "content": system_prompt})
        for i, content in enumerate(example):
            content = content.strip()
            if content.startswith("user:"):
                dialog.append({"role": "user", "content": content[5:].strip()})
            elif content.startswith("assistant:"):
                dialog.append({"role": "assistant", "content": content[10:].strip()})
        return {"messages": dialog}

def create(
    dataset_path: str = "jaggery-ai/conversations", 
    split: str = "train", 
    system_prompt: str = "You are Jaggery an expert counselling psychologist. You are talking to a client. Be empathetic and make them feel better."
):
    """Creates the dataset from the given path."""
    dataset = datasets.load_dataset(path=dataset_path, trust_remote_code=True, split=split)
    conversations = list()
    for data in tqdm(dataset, desc="Processing dataset", total=len(dataset)):
        conversations.append(to_dialog(data["text"], system_prompt))
    return conversations

# convert conversations to jsonl format
def to_jsonl(conversations: list, output_file: str):
    with open(output_file, "w") as f:
        for conversation in conversations:
            json.dump(conversation, f)
            f.write(f"\n")
        print(f"Saved to {output_file}")


if __name__ == '__main__':
    # create(dataset_path='jaggery-ai/conversations')

    splits = ['train', 'validation']

    for split in splits:
        conversations = create(dataset_path=os.path.abspath('../../jaggery_dataset'), split=split)
        print(f"Total {split} conversations: {len(conversations)}")
        to_jsonl(conversations, f'data/{split}.jsonl')


    



