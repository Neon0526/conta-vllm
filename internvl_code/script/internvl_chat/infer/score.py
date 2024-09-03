import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Load predictions data
with open('/home/new/桌面/intern/InternVL/internvl_chat/infer/epoch5/normal/bgr2_output_new.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize lists to store BLEU and ROUGE scores
bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Initialize smoothing function for BLEU score
smooth_func = SmoothingFunction().method1

for item in data:
    question = item['question']
    standard_answer = item['standard_answer']
    predicted_answer = item['response']
    
    # Tokenize answers for BLEU scoring
    actual_answer_tokens = standard_answer.split()
    predicted_answer_tokens = predicted_answer.split()
    
    # Compute BLEU score with smoothing
    bleu_score = sentence_bleu([actual_answer_tokens], predicted_answer_tokens, smoothing_function=smooth_func)
    bleu_scores.append(bleu_score)
    
    # Compute ROUGE scores
    rouge_scores = scorer.score(standard_answer, predicted_answer)
    rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
    rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
    rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

# Calculate average BLEU and ROUGE scores
average_bleu = sum(bleu_scores) / len(bleu_scores)
average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

# Print results
print(f'Average BLEU Score: {average_bleu:.4f}')
print(f'Average ROUGE-1 Score: {average_rouge1:.4f}')
print(f'Average ROUGE-2 Score: {average_rouge2:.4f}')
print(f'Average ROUGE-L Score: {average_rougeL:.4f}')
