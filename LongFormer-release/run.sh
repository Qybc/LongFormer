# Longformer for ADNI
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 26521 main.py --dataset_file ADNI --train_data_path ./local_data/ADNI-train-sub.csv --test_data_path ./local_data/ADNI-test-sub.csv --batch_size 2 --num_feature_scales 1 --n_times 2 --hidden_dim 256 --num_classes 2 --vision_encoder densenet121 --nheads 8 --num_queries 125 --distributed


# Longformer for OASIS
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 26621 main.py --dataset_file OASIS --train_data_path ./local_data/oasis_hong_train.csv --test_data_path ./local_data/oasis_hong_test.csv --batch_size 4 --num_feature_scales 1 --n_times 1 --hidden_dim 256 --num_classes 2 --vision_encoder densenet121 --nheads 8 --num_queries 125 --distributed

# Longformer for AIBL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 26621 main.py --dataset_file AIBL --train_data_path ./local_data/AIBL-train-sub.csv --test_data_path ./local_data/AIBL-test.csv --batch_size 4 --num_feature_scales 1 --n_times 1 --hidden_dim 256 --num_classes 2 --vision_encoder densenet121 --nheads 8 --num_queries 125 --distributed



# Baseline: Res50 for OASIS
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29021 baseline.py --dataset OASIS --train_data_path ./local_data/oasis_hong_train.csv --test_data_path ./local_data/oasis_hong_test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res50 --distributed

# Baseline: Res101 for OASIS
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29121 baseline.py --dataset OASIS --train_data_path ./local_data/oasis_hong_train.csv --test_data_path ./local_data/oasis_hong_test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res101 --distributed

# Baseline: Res152 for OASIS
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29221 baseline.py --dataset OASIS --train_data_path ./local_data/oasis_hong_train.csv --test_data_path ./local_data/oasis_hong_test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res152 --distributed

# Baseline: MedicalNet for OASIS
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29321 baseline.py --dataset OASIS --train_data_path ./local_data/oasis_hong_train.csv --test_data_path ./local_data/oasis_hong_test.csv --batch_size 2 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model medicalnet --distributed

# Baseline: Res50 for ADNI-subset
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29421 baseline.py --dataset ADNI --train_data_path ./local_data/ADNI-train.csv --test_data_path ./local_data/ADNI-test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res50 --distributed

# Baseline: Res101 for ADNI-subset
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29621 baseline.py --dataset ADNI --train_data_path ./local_data/ADNI-train.csv --test_data_path ./local_data/ADNI-test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res101 --distributed

# Baseline: Res152 for ADNI
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29521 baseline.py --dataset ADNI --train_data_path ./local_data/ADNI-train.csv --test_data_path ./local_data/ADNI-test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res152 --distributed

# Baseline: densenet121 for ADNI
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port 28021 baseline.py --dataset ADNI --train_data_path ./local_data/ADNI-train.csv --test_data_path ./local_data/ADNI-test.csv --batch_size 4 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model densenet121 --distributed

# Baseline: medicalnet for ADNI
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port 28121 baseline.py --dataset ADNI --train_data_path ./local_data/ADNI-train.csv --test_data_path ./local_data/ADNI-test.csv --batch_size 2 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model medicalnet --distributed

# Baseline: vit for ADNI
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1  --master_port 28221 baseline.py --dataset ADNI --train_data_path ./local_data/ADNI-train.csv --test_data_path ./local_data/ADNI-test.csv --batch_size 4 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model vit --distributed


# Baseline: Res50 for AIBL
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  --master_port 27021 baseline.py --dataset AIBL --train_data_path ./local_data/AIBL-train.csv --test_data_path ./local_data/AIBL-test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res50 --distributed

# Baseline: Res101 for AIBL
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1  --master_port 27121 baseline.py --dataset AIBL --train_data_path ./local_data/AIBL-train.csv --test_data_path ./local_data/AIBL-test.csv --batch_size 2 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res101 --distributed

# Baseline: Res152 for AIBL
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_port 27221 baseline.py --dataset AIBL --train_data_path ./local_data/AIBL-train.csv --test_data_path ./local_data/AIBL-test.csv --batch_size 6 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model res152 --distributed

# Baseline: densenet121 for AIBL
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port 28021 baseline.py --dataset AIBL --train_data_path ./local_data/AIBL-train.csv --test_data_path ./local_data/AIBL-test.csv --batch_size 4 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model densenet121 --distributed

# Baseline: medicalnet for AIBL
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1  --master_port 27021 baseline.py --dataset AIBL --train_data_path ./local_data/AIBL-train.csv --test_data_path ./local_data/AIBL-test.csv --batch_size 2 --num_feature_scales 3 --n_times 1 --hidden_dim 768 --num_classes 2 --model medicalnet --distributed