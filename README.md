# vllm_helper

Helper scripts and examples for running **vLLM** on Stanford clusters (Sherlock, Marlowe, Yens) and for evaluating **fine-tuned LoRA adapters** trained on [Together](https://together.ai).

This repo accompanies the blog post [Fine-Tuning Open Source Models with Together + vLLM](link).  
It provides the **“try it yourself”** walkthrough: from preparing JSONL datasets to running base and fine-tuned models on Sherlock.

---
In this example, we fine-tune Qwen3-8B-Base to classify Reddit posts into one of ten subreddits. With no fine-tuning, the base model reached an accuracy of 0.39 on our test set. After fine-tuning with LoRA adapters, accuracy nearly doubled to 0.74.

We’ll walk step by step through:

1. Preparing training data in JSONL format

2. Submitting a fine-tuning job to Together

3. Downloading the LoRA adapter

4. Running both the base and fine-tuned models locally with vLLM


## Step 1. Define the Task and Dataset

Our task is: given the title and body of a Reddit post, predict which subreddit it belongs to.

- Input: title + body

- Output: subreddit name (one of ten choices)

We prepared a dataset with:

- Training set: 9,800 examples per class
- Validation set: 100 examples per class
- Test set: 500 examples per class

Each row is stored as JSONL (JSON Lines). Together expects the following format:

```{.yaml .no-copy title="Expected JSONL format"}
{"prompt": "Post title\n\nPost body", "completion": "subreddit_name"}
```
Each row is stored as JSONL (JSON Lines). Together expects the following format:

```{.yaml .no-copy title="Expected JSONL format"}
{"prompt": "Post title\n\nPost body", "completion": "subreddit_name"}
```

This structure is all you need — one input string, one output string per line.

## Step 2. Set Up the Environment
On Sherlock we created a Python environment for data prep and training:
```bash title="Terminal Input From Login Node"
cd <project-space>/llm-ft
ml python/3.12.1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This ensures we can build JSONL files, interact with Together’s API, and later run inference locally.
## Step 3. Upload Training Files to Together
With train.jsonl, val.jsonl and test.jsonl ready, we upload the training and (optionally) validation sets:
```bash title="Terminal Input From Login Node"
together files upload train.jsonl
together files upload val.jsonl
```

## Step 4. Run a Fine-Tuning Job
You can launch the fine-tuning job via Together CLI or through the web interface.

The web interface makes it easy to adjust parameters, track progress, and view checkpoints.

Login to [Together](https://api.together.xyz/fine-tuning){target="_blank"} and go to Fine-tuning tab to start a new job.

First, select the base model.

We chose `Qwen/Qwen3-8B-Base` model.

You’ll be prompted to select parameters. For our experiment, we chose the following:

First, select the base model.

We chose `Qwen/Qwen3-8B-Base` model.

You’ll be prompted to select parameters. For our experiment, we chose the following:

- Epochs: 1

- Checkpoints: 1

- Evaluations: 4

- Batch size: 8

- LoRA rank: 32

- LoRA alpha: 64

- LoRA dropout: 0.05

- LoRA trainable modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

- Train on inputs: false

- Learning rate: 0.0001

- Learning rate scheduler: cosine

- Warmup ratio: 0.06

- Scheduler cycles: 3

- Max gradient norm: 1

- Weight decay: 0.01

You can optionally connect to your [Weights & Biases](https://wandb.ai/home) project to track training and validation losses graphically.

You can experiment with different values depending on your dataset size and task.

When the job completes, you’ll be able to download the resulting LoRA adapter checkpoint (or the merged model). This adapter contains only the learned weights from fine-tuning — a lightweight file we’ll use in combination with the base model.

This training cost $5 and ran in 11 minutes.

## Step 5. Download the trained LoRA Adapter

After training is finished, download the LoRA adapter and copy to your project space on Sherlock.

In this case, we made a models directory in our project space and copy the adapter to `<project-space>/llm-ft/models/qwen3-8b-1epoch-10k-data-32-lora`.
