
dataset_name='PBMC4k'
python get_cell_emb.py \
  --input_data data/${dataset_name}.h5ad \
  --dataset_name ${dataset_name} \
  --output_dir results/cell_embedding \
  --n_top_genes 0 \
  --early_stopping \
  --verbose
