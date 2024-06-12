topk=4
mkdir -p ./results/factoidqa
mkdir -p ./results/strategyqa
mkdir -p ./results/entityquestions
for file in *.jsonl; do
    if [[ $file == factoidqa_* ]]; then
        new_file="${file#factoidqa_dev_}"
        mv "$file" "./results/factoidqa/$new_file"
        
    fi
    if [[ $file == strategyqa_* ]]; then
        new_file="${file#strategyqa_}"
        mv "$file" "./results/strategyqa/$new_file"
    fi
    if [[ $file == entityquestions_* ]]; then
        new_file="${file#entityquestions_}"
        mv "$file" "./results/entityquestions/$new_file"
    fi
done
cd results/factoidqa
for file in *.jsonl; do
    if [[ $file == hfllm_closed_book_* ]]; then
        new_dir="Closed-Book"
        mkdir -p "$new_dir"
        new_file="${file#hfllm_closed_book_}"
        mv "$file" "$new_dir/${new_file/_max_gen_10/}"
    fi
    if [[ $file == hfllm_open_book_ANCE_100_topk_${topk}_* ]]; then
        new_dir="ANCE"
        mkdir -p "$new_dir"
        new_file="${file#hfllm_open_book_ANCE_100_topk_${topk}_}"
        mv "$file" "$new_dir/${new_file/_max_gen_10/}"
    fi
    if [[ $file == hfllm_open_book_BM25_100_topk_${topk}_* ]]; then
        new_dir="BM25"
        mkdir -p "$new_dir"
        new_file="${file#hfllm_open_book_BM25_100_topk_${topk}_}"
        mv "$file" "$new_dir/${new_file/_max_gen_10/}"
    fi
    if [[ $file == hfllm_open_book_DPR_100_topk_${topk}_* ]]; then
        new_dir="DPR"
        mkdir -p "$new_dir"
        new_file="${file#hfllm_open_book_DPR_100_topk_${topk}_}"
        mv "$file" "$new_dir/${new_file/_max_gen_10/}"
    fi
    for n in 50 100 300 1000; do
        if [[ $file == hfllm_open_book_Oracle_${n}_topk_${topk}_* ]]; then
            new_dir="Oracle${n}"
            mkdir -p "$new_dir"
            new_file="${file#hfllm_open_book_Oracle_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/${new_file/_max_gen_10/}"
        fi
        if [[ $file == hfllm_open_book_SpEL_${n}_topk_${topk}_* ]]; then
            new_dir="SpEL${n}"
            mkdir -p "$new_dir"
            new_file="${file#hfllm_open_book_SpEL_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/${new_file/_max_gen_10/}"
        fi
    done
done
cd ../strategyqa
for file in *.jsonl; do
    if [[ $file == dev_hfllm_closed_book_* ]]; then
        new_dir="Closed-Book"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_closed_book_}"
        mv "$file" "$new_dir/train_filtered_${new_file/_max_gen_1/}"
    fi
    if [[ $file == train_hfllm_closed_book_* ]]; then
        new_dir="Closed-Book"
        mkdir -p "$new_dir"
        new_file="${file#train_hfllm_closed_book_}"
        mv "$file" "$new_dir/train_${new_file/_max_gen_1/}"
    fi
    if [[ $file == dev_hfllm_open_book_ANCE_100_topk_${topk}_* ]]; then
        new_dir="ANCE"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_open_book_ANCE_100_topk_${topk}_}"
        mv "$file" "$new_dir/train_filtered_${new_file/_max_gen_1/}"
    fi
    if [[ $file == train_hfllm_open_book_ANCE_100_topk_${topk}_* ]]; then
        new_dir="ANCE"
        mkdir -p "$new_dir"
        new_file="${file#train_hfllm_open_book_ANCE_100_topk_${topk}_}"
        mv "$file" "$new_dir/train_${new_file/_max_gen_1/}"
    fi
    if [[ $file == dev_hfllm_open_book_BM25_100_topk_${topk}_* ]]; then
        new_dir="BM25"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_open_book_BM25_100_topk_${topk}_}"
        mv "$file" "$new_dir/train_filtered_${new_file/_max_gen_1/}"
    fi
    if [[ $file == train_hfllm_open_book_BM25_100_topk_${topk}_* ]]; then
        new_dir="BM25"
        mkdir -p "$new_dir"
        new_file="${file#train_hfllm_open_book_BM25_100_topk_${topk}_}"
        mv "$file" "$new_dir/train_${new_file/_max_gen_1/}"
    fi
    if [[ $file == dev_hfllm_open_book_DPR_100_topk_${topk}_* ]]; then
        new_dir="DPR"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_open_book_DPR_100_topk_${topk}_}"
        mv "$file" "$new_dir/train_filtered_${new_file/_max_gen_1/}"
    fi
    if [[ $file == train_hfllm_open_book_DPR_100_topk_${topk}_* ]]; then
        new_dir="DPR"
        mkdir -p "$new_dir"
        new_file="${file#train_hfllm_open_book_DPR_100_topk_${topk}_}"
        mv "$file" "$new_dir/train_${new_file/_max_gen_1/}"
    fi
    for n in 50 100 300 1000; do
        if [[ $file == dev_hfllm_open_book_Oracle_${n}_topk_${topk}_* ]]; then
            new_dir="Oracle${n}"
            mkdir -p "$new_dir"
            new_file="${file#dev_hfllm_open_book_Oracle_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/train_filtered_${new_file/_max_gen_1/}"
        fi
        if [[ $file == dev_hfllm_open_book_SpEL_${n}_topk_${topk}_* ]]; then
            new_dir="SpEL${n}"
            mkdir -p "$new_dir"
            new_file="${file#dev_hfllm_open_book_SpEL_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/train_filtered_${new_file/_max_gen_1/}"
        fi
        if [[ $file == train_hfllm_open_book_Oracle_${n}_topk_${topk}_* ]]; then
            new_dir="Oracle${n}"
            mkdir -p "$new_dir"
            new_file="${file#train_hfllm_open_book_Oracle_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/train_${new_file/_max_gen_1/}"
        fi
        if [[ $file == train_hfllm_open_book_SpEL_${n}_topk_${topk}_* ]]; then
            new_dir="SpEL${n}"
            mkdir -p "$new_dir"
            new_file="${file#train_hfllm_open_book_SpEL_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/train_${new_file/_max_gen_1/}"
        fi
    done
done
cd ../entityquestions
for file in *.jsonl; do
  if [[ $file == dev_hfllm_closed_book_* ]]; then
        new_dir="Closed-Book"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_closed_book_}"
        mv "$file" "$new_dir/dev_${new_file/_max_gen_10/}"
    fi
    if [[ $file == test_hfllm_closed_book_* ]]; then
        new_dir="Closed-Book"
        mkdir -p "$new_dir"
        new_file="${file#test_hfllm_closed_book_}"
        mv "$file" "$new_dir/test_${new_file/_max_gen_10/}"
    fi
    if [[ $file == dev_hfllm_open_book_ANCE_100_topk_${topk}_* ]]; then
        new_dir="ANCE"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_open_book_ANCE_100_topk_${topk}_}"
        mv "$file" "$new_dir/dev_${new_file/_max_gen_10/}"
    fi
    if [[ $file == test_hfllm_open_book_ANCE_100_topk_${topk}_* ]]; then
        new_dir="ANCE"
        mkdir -p "$new_dir"
        new_file="${file#test_hfllm_open_book_ANCE_100_topk_${topk}_}"
        mv "$file" "$new_dir/test_${new_file/_max_gen_10/}"
    fi
    if [[ $file == dev_hfllm_open_book_BM25_100_topk_${topk}_* ]]; then
        new_dir="BM25"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_open_book_BM25_100_topk_${topk}_}"
        mv "$file" "$new_dir/dev_${new_file/_max_gen_10/}"
    fi
    if [[ $file == test_hfllm_open_book_BM25_100_topk_${topk}_* ]]; then
        new_dir="BM25"
        mkdir -p "$new_dir"
        new_file="${file#test_hfllm_open_book_BM25_100_topk_${topk}_}"
        mv "$file" "$new_dir/test_${new_file/_max_gen_10/}"
    fi
    if [[ $file == dev_hfllm_open_book_DPR_100_topk_${topk}_* ]]; then
        new_dir="DPR"
        mkdir -p "$new_dir"
        new_file="${file#dev_hfllm_open_book_DPR_100_topk_${topk}_}"
        mv "$file" "$new_dir/dev_${new_file/_max_gen_10/}"
    fi
    if [[ $file == test_hfllm_open_book_DPR_100_topk_${topk}_* ]]; then
        new_dir="DPR"
        mkdir -p "$new_dir"
        new_file="${file#test_hfllm_open_book_DPR_100_topk_${topk}_}"
        mv "$file" "$new_dir/test_${new_file/_max_gen_10/}"
    fi
    for n in 50 100 300 1000; do
        if [[ $file == dev_hfllm_open_book_Oracle_${n}_topk_${topk}_* ]]; then
            new_dir="Oracle${n}"
            mkdir -p "$new_dir"
            new_file="${file#dev_hfllm_open_book_Oracle_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/dev_${new_file/_max_gen_10/}"
        fi
        if [[ $file == dev_hfllm_open_book_SpEL_${n}_topk_${topk}_* ]]; then
            new_dir="SpEL${n}"
            mkdir -p "$new_dir"
            new_file="${file#dev_hfllm_open_book_SpEL_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/dev_${new_file/_max_gen_10/}"
        fi
        if [[ $file == test_hfllm_open_book_Oracle_${n}_topk_${topk}_* ]]; then
            new_dir="Oracle${n}"
            mkdir -p "$new_dir"
            new_file="${file#test_hfllm_open_book_Oracle_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/test_${new_file/_max_gen_10/}"
        fi
        if [[ $file == test_hfllm_open_book_SpEL_${n}_topk_${topk}_* ]]; then
            new_dir="SpEL${n}"
            mkdir -p "$new_dir"
            new_file="${file#test_hfllm_open_book_SpEL_${n}_topk_${topk}_}"
            mv "$file" "$new_dir/test_${new_file/_max_gen_10/}"
        fi
    done
done
cd ..

current_date=$(date +"%Y%m%d")
zip -r experimental_results_${current_date}.zip factoidqa entityquestions strategyqa
