cd ..
settings=('Closed-Book' 'DPR' 'RePLUG' 'EntityRetrieval-Oracle' 'EntityRetrieval-SpEL')

for setting in "${settings[@]}"; do
  python3.10 main.py experimental-results-path="results/FactoidQA/${setting}"
done