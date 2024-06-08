cd ..
settings=('Closed-Book' 'ANCE' 'BM25' 'DPR' 'SpEL50' 'SpEL100' 'SpEL300' 'SpEL1000' 'Oracle50' 'Oracle100' 'Oracle300' 'Oracle1000')

for setting in "${settings[@]}"; do
  python3.10 main.py experimental-results-path="results/EntityQuestions/1/${setting}"
done