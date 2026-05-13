import argparse
import torch
import numpy as np
import random


def get_config():
    parser = argparse.ArgumentParser(description='Multi-modal Drug-miRNA Association Prediction')

    # 文件路径配置
    parser.add_argument('--link_file', type=str, default='../data/miRNA_drug.csv',
                        help='Known associations file (pairs)')

    # miRNA 相关文件
    parser.add_argument('--mirna_file', type=str, default='../data/miRNA.csv',
                        help='miRNA names file')
    parser.add_argument('--mirna_similarity', type=str, default='../data/miRNA_similarity.csv',
                        help='miRNA similarity matrix')
    parser.add_argument('--mirna_kmer', type=str, default='../feature/rna_fea/miRNA_kmer.csv',
                        help='miRNA k-mer features')
    parser.add_argument('--mirna_doc2vec', type=str, default='../feature/rna_fea/miRNA_doc2vec_features.csv',
                        help='miRNA doc2vec features')
    parser.add_argument('--mirna_rnafm', type=str, default='../feature/rna_fea/miRNA_fm640.csv',
                        help='miRNA RNA-FM features')
    parser.add_argument('--mirna_des_embedding', type=str, default='../feature/miRNA_embeddings_rag.csv',
                        help='miRNA description embeddings')

    # Drug 相关文件
    parser.add_argument('--drug_file', type=str, default='../data/drug.csv',
                        help='Drug names file')
    parser.add_argument('--drug_similarity', type=str, default='../data/drug_similarity.csv',
                        help='Drug similarity matrix')
    parser.add_argument('--drug_gin_features', type=str, default='../feature/drug_fea/drug_GIN_64.csv',
                        help='Drug GIN features')
    parser.add_argument('--drug_MACCS', type=str, default='../feature/drug_fea/drug_MACCS.csv',
                        help='Drug MACCS fingerprint')
    parser.add_argument('--drug_atom_feature', type=str, default='../feature/drug_fea/drug_atom_features.csv',
                        help='Drug atom features')
    parser.add_argument('--drug_des_embedding', type=str, default='../feature/drug_embeddings_rag.csv',
                        help='Drug description embeddings')

    # 特征维度配置
    # miRNA特征维度
    parser.add_argument('--doc2vec_dim', type=int, default=100,
                        help='miRNA doc2vec feature dimension')
    parser.add_argument('--kmer_dim', type=int, default=64,
                        help='miRNA k-mer feature dimension')
    parser.add_argument('--rnafm_dim', type=int, default=640,
                        help='miRNA RNA-FM feature dimension')
    parser.add_argument('--mirna_semantic_dim', type=int, default=384,
                        help='miRNA semantic embedding dimension')

    # Drug特征维度
    parser.add_argument('--atom_dim', type=int, default=85,
                        help='Drug atom feature dimension')
    parser.add_argument('--gin_dim', type=int, default=64,
                        help='Drug GIN feature dimension')
    parser.add_argument('--maccs_dim', type=int, default=167,
                        help='Drug MACCS fingerprint dimension')
    parser.add_argument('--drug_semantic_dim', type=int, default=384,
                        help='Drug semantic embedding dimension')

    # 实体数量
    parser.add_argument('--miRNA_numbers', type=int, default=701,
                        help='Number of miRNAs')
    parser.add_argument('--drug_numbers', type=int, default=101,
                        help='Number of drugs')


    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Unified embedding dimension (core dimension for all modules)')
    parser.add_argument('--pro_dim', type=int, default=256,
                        help='GCN output dimension - single GCN processes fused 4 features')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for HeteroGNN intermediate layers (output feature dimension)')

    # 网络结构参数
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of HeteroGNN layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads in MCF Transformer (must divide embedding_dim)')

    # MLP参数
    parser.add_argument('--outSize', type=int, default=1,
                        help='MLP output dimension (1 for binary prediction)')

    # ==================== 训练超参数 ====================
    parser.add_argument('--kfold', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--batchSize', type=int, default=256,
                        help='Batch size (increased for better training stability)')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Maximum number of epochs (early stopping active)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for Adam optimizer (reduced from 1e-2 for stability)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 regularization weight decay')
    parser.add_argument('--fcDropout', type=float, default=0.5,
                        help='Dropout rate for all layers (attention, FC, etc.)')


    # 其他配置
    parser.add_argument('--similarity_threshold', type=float, default=0.5,
                        help='Similarity threshold for graph construction (range: 0.0-1.0, lower=more edges)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()


    # 参数验证和初始化
    if args.embedding_dim % args.num_heads != 0:
        raise ValueError(f"embedding_dim ({args.embedding_dim}) must be divisible by num_heads ({args.num_heads})")

    if args.pro_dim != args.embedding_dim:
        print(f"Warning: pro_dim ({args.pro_dim}) != embedding_dim ({args.embedding_dim})")
        print(f"Recommend setting pro_dim = embedding_dim for consistency")

    if args.lr > 1e-2:
        print(f"Warning: Learning rate {args.lr} might be too high for Adam optimizer")
        print(f"Recommended range: 1e-4 to 1e-3")

    if args.similarity_threshold > 0.7:
        print(f"Warning: similarity_threshold={args.similarity_threshold} is high, may result in sparse graphs")
        print(f"Consider using 0.3-0.5 for better connectivity")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    return args
