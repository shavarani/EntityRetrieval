datasets=('FACTOIDQA' 'STRATEGYQA')
splits=('train' 'dev')
index_name="exact"
model_type="single-nq"
k=100

for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    output_file="output_${dataset}_${split}_${index_name}_${model_type}_${k}.jsonl"
    python create_dpr_context.py --dataset "$dataset" --split "$split" --index_name "$index_name" --question_model_type "$model_type" --retriever_k "$k" --output_file "$output_file"
  done
done