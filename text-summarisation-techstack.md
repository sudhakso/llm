# Text Summarisation

The doccument provides a summary of techstack required to build a text summarisation application using LLM.

Disclaimer: The approach and code is inspired from Taughdata tutorial

## Tools

* Hugging Face: A platform that provides access to the FLAN-T5 model, facilitating its download and usage for fine-tuning

* Transformers: This is used to simplify the process of loading the pre-trained FLAN-T5 model and provides useful functions for fine-tuning

* Datasets: a collection of ready-to-use datasets, crucial for sourcing relevant data for fine-tuning

* Sentencepiece: a tokenization tool

* Tokenizers: Tokenizer library for converting text into a tokens and map it to vectors. We will use T5 tokenizer

* Evaluate: Library that provides wide range pf metrics to assess the performance

* Rouge score: specific metric to evaluate quality of text generated

* NLTK: used for text cleaning and stemming

## Getting Started (Setup)

Open a Jupyter notebook, and run the following commands

```
%%bash
pip install nltk
pip install datasets
pip install transformers[torch]
pip instrall tokenizers
pip install evaluate
pip install rouge_score
pip install sentencepiece
pip install huggingface_hub 
```

Include these libraries into your notebook

```
import nltk
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

```
Next is the choice of the FLAN-T5 variant to use that fits the purpose, FLAN-T5 comes in following variants,

* google/flan-t5-small: 80m parameters, 300MB memory
* google/flan-t5-base: 250m parameters, 990MB
* google/flan-t5-large: 780m parameters, 1GB
* google/flan-t5-xl: 3B parameters, 12GB
* google/flan-t5-xxl: 11B parameters, 80GB

If you have a GPU in your system google/flan-t5-base provides an optimal choice with a  balance on computing power and performance.

For the experiment, we will choose flan-t5-small. We shall also choose an appropriate Tokenizer based on the choice of the model.

Apart from the model and tokenizer, we need a module that will produce the question-answering task.

There are few choices here, and we will continue with DataCollatorForSe2Seq for Q&A modelling. 

```
MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collector = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

```

Any Encoder-Decoder model training undergoes following steps,
* Tokenizing the input
* Generating embedding vector for input sequence
* Encoding with self-attention layers
* Decoding step to generate an output
* Softmax layer at output
* Output is fed back into Decoding layer

The above steps are repeated until we get End of token string in the input sequence.

The steps below loads the data, we use yahoo Q&A data. There are other sample datasets available and one can choose the best dataset to fit the use case.

```
DATA_NAME = "yahoo_answers_qa"
yahoo_answers_qa = load_dataset(DATA_NAME)
```

Next, we split the data into Train & Test sets. A typical proportion is 70:30 between train:test

```
yahoo_answers_qa = yahoo_answers_qa["train"].train_test_split(test_size=0.3)
```

Next, step is to perform Data formatting & tokenization. The process of data formatting is called IHT - Instruction Fine Tuning. The process improves model performance significantly on unseen data.

You can learn more on [IHT-finetuning](https://www.toughdata.net/blog/post/finetune-flan-t5-question-answer-quora-dataset)

As described in the article above, We will prefix our instructions with "*Please answer this question*:Question". As part of the preprocessing, we will transform our input data in the Q&A format, 

```
prefix = "Please answer this question: "

def preprocess_function(examples):
   inputs = [prefix + doc for doc in examples["question"]]
   model_inputs = tokenizer(inputs, max_length=128, truncation=True)

   labels = tokenizer(text_target=examples["answer"],
                      max_length=512,
                      truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Run a map function to transform the input dataset
tokenized_dataset = yahoo_answers_qa.map(preprocess_function, batched=True)

```

Next, is to determine the training metrics that will be used to evaluate the performance of a text generation model. There are 2 prominent scores,
* BLEU
* ROUGE

Both techniques generates score by comparing it to a reference answer. We will stick to ROUGE score and keep BLEU for another writeup.

Higher the ROUGE score better the performance. Ideal score is >0.7.

Next, we will setup ROUGE metric evaluator

```
nltk.download("punkt")
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  
   return result
```

Into the training process using IHT.... 

For IHT, like any other training we need to set some hyperparameters, here are the main ones,
* Learning rate: The value decides how quickly the model converge. Will choose 2e-5 for our experiment.
* Batch size: Number of sample for an iteration. A typical value is between 8 - 10. We will choose 8 for our experiment.
* Number of epochs: Determines the total number of passes through the training dataset. We will choose 3 passes.

```
# Global Parameters
L_RATE = 2e-5
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="./results",
   evaluation_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)
```
Next, tets instantiate a trainer instance to manage the overall training process,

```
trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)
```

Next, start the training process for the fine-tuning the T5 model

```
trainer.train()
```
We must monitor the rouge metrics for each epochs. And the metric value should increase else we must retune the hyperparameters or add new dataset samples.

Output will be generated in *results* folder. The checkpoints can be loaded for inference.

## Inference

* 1'st step is to load the fine-tinel model from the last checkpoint,
```
last_checkpoint = "./results/checkpoint-21500"

finetuned_model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
tokenizer = T5Tokenizer.from_pretrained(last_checkpoint)
```

* 2'nd step is to prepare a Question to ask,
```
my_question = "What platform is best suited for federated learning"
inputs = "Please answer to this question: " + my_question
```

* 3'rd step is to run the prediction (this needs to be included in the chat app)

```
inputs = tokenizer(inputs, return_tensors="pt")
outputs = finetuned_model.generate(**inputs)
answer = tokenizer.decode(outputs[0])

print(answer)
```
The code explained in this blog is available here [FLAN_T5_Finetuning_QA_Yahoo_Data.ipynb](https://github.com/sudhakso/llm/blob/main/FLAN_T5_Finetuning_QA_Yahoo_Data.ipynb)

