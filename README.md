# Intent-Detection
Intent Detection with DistilBERT (Fine-Tuning)  
This project shows how to fine-tune DistilBERT, a smaller and faster version of BERT, for classifying user intent. This is useful in chatbots and natural language understanding.

Overview  
Intent detection helps understand what a user wants based on their input, like identifying &quot;get_weather&quot; for the question &quot;What's the weather like today?&quot;  
We fine-tune a pre-trained DistilBERT on a labeled dataset of text and intent pairs to do this efficiently.

Project Structure  
```
intent-detection-distilbert/
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── labels.txt
├── model/                      # Saved model checkpoints
├── src/
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── train.py                # Fine-tuning script
│   ├── predict.py              # Inference script
├── utils/
│   └── metrics.py              # Accuracy, F1, etc.
├── intent_detection.ipynb      # Training and evaluation notebook
└── README.md
```

Requirements  
- Python 3.7+  
- Transformers (HuggingFace)  
- PyTorch  
- scikit-learn  
- pandas  
- tqdm  

Install dependencies:  
```bash
pip install -r requirements.txt
```

Dataset Format  
The dataset should be a CSV with two columns:  
```
text,label
&quot;Book a flight to New York&quot;,&quot;book_flight&quot;
&quot;What is the weather in Paris?&quot;,&quot;get_weather&quot;
```  
The labels.txt file lists all intent classes, one per line.

Training the Model  
```bash
python src/train.py \
  --train_file data/train.csv \
  --val_file data/val.csv \
  --model_name distilbert-base-uncased \
  --output_dir model/ \
  --batch_size 16 \
  --epochs 4
```

Inference  
To predict the intent of a sentence:  
```bash
python src/predict.py \
  --model_dir model/ \
  --sentence &quot;Show me nearby restaurants&quot;
```  
This will output the predicted intent, like &quot;find_restaurant.&quot;

Evaluation  
Evaluation uses these metrics:  
- Accuracy  
- Precision and Recall  
- F1 Score (macro and weighted)  

Use metrics.py or the notebook to check performance.

Use Cases  
- Chatbots (customer service)  
- Voice assistants  
- Command recognition in apps

References  
- DistilBERT paper  
- HuggingFace Transformers docs  
- Public intent datasets like CLINC150, SNIPS
