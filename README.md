# GPT-2 Fine-Tuned Text Generation App

## Project Overview
This project demonstrates how a pre-trained transformer-based language model (GPT-2) can be fine-tuned on a custom dataset to generate coherent and contextually relevant text. The fine-tuned model is further deployed as an interactive web application, making it easy to showcase and interact with the text generation system.

## Objective
The objective of this project is to adapt a pre-trained GPT-2 model to a specific domain using transfer learning, generate meaningful text from user prompts, and present the model through a simple and user-friendly web interface.

## About GPT-2
GPT-2 (Generative Pre-trained Transformer 2) is a transformer-based language model developed by OpenAI. It is trained on a large corpus of internet text and is capable of generating human-like text by predicting the next word in a sequence.

## What is Fine-Tuning
Fine-tuning is the process of training a pre-trained model on a smaller, task-specific dataset. Instead of learning language from scratch, the model adjusts its existing knowledge to better match the style and structure of the new data. In this project, GPT-2 is fine-tuned to generate text similar to the custom training dataset.

## Dataset
The dataset used in this project is a plain text file (`train.txt`) where each line represents a sentence or paragraph. The dataset is stored inside a `data` directory and is used to train the GPT-2 model using self-supervised learning.

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Streamlit
- Visual Studio Code

## Training Process
The training process starts by loading a pre-trained GPT-2 model and tokenizer. The dataset is tokenized and prepared for causal language modeling, where input tokens are also used as labels. The model is fine-tuned for multiple epochs using transfer learning and saved locally after training.

## Text Generation
After training, the fine-tuned model can generate text by taking a user-provided prompt and predicting the next sequence of words. The generated text is coherent, context-aware, and reflects the style of the training data.

## Web Application
The trained model is deployed using Streamlit. The web application allows users to enter a prompt, adjust text generation parameters, and generate text interactively in real time through a browser interface.

## How to Run the Project

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Streamlit
- Visual Studio Code

## Training Process
The training process starts by loading a pre-trained GPT-2 model and tokenizer. The dataset is tokenized and prepared for causal language modeling, where input tokens are also used as labels. The model is fine-tuned for multiple epochs using transfer learning and saved locally after training.

## Text Generation
After training, the fine-tuned model can generate text by taking a user-provided prompt and predicting the next sequence of words. The generated text is coherent, context-aware, and reflects the style of the training data.

## Web Application
The trained model is deployed using Streamlit. The web application allows users to enter a prompt, adjust text generation parameters, and generate text interactively in real time through a browser interface.

## How to Run the Project

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Streamlit
- Visual Studio Code

## Training Process
The training process starts by loading a pre-trained GPT-2 model and tokenizer. The dataset is tokenized and prepared for causal language modeling, where input tokens are also used as labels. The model is fine-tuned for multiple epochs using transfer learning and saved locally after training.

## Text Generation
After training, the fine-tuned model can generate text by taking a user-provided prompt and predicting the next sequence of words. The generated text is coherent, context-aware, and reflects the style of the training data.

## Web Application
The trained model is deployed using Streamlit. The web application allows users to enter a prompt, adjust text generation parameters, and generate text interactively in real time through a browser interface.

## How to Run the Project

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate

###step 2 : install dependencies 
pip install -r requirements.txt

###Step 3: Train the Model
python train.py

###generate text
python generate.py

###run the web app
streamlit run app.py

Results

The fine-tuned GPT-2 model successfully generates coherent and contextually relevant text. The output aligns with the style and structure of the custom dataset, and the web application provides an effective way to demonstrate the model’s capabilities.

Author

Sneha S. Pawar