import pandas as pd
import torch
from torch import nn
from datalord import load_sim_edge_index, load_edge_ws_index
from init_model import GCNFeatureExtractor, HeteroGNN, ResidualPredictor
from BCQE_CA import MultiModalCCABlock

class MSFEICL(nn.Module):

    def __init__(self, args):
        super(MSFEICL, self).__init__()
        self.args = args
        device = args.device

        # miRNA特征
        self.miRNA_doc2vec = pd.read_csv(args.mirna_doc2vec, header=None).values
        self.miRNA_kmer = pd.read_csv(args.mirna_kmer, header=None).values
        self.miRNA_rnafm = pd.read_csv(args.mirna_rnafm, header=None).values
        self.miRNA_sim = pd.read_csv(args.mirna_similarity, header=0).iloc[:, 1:].values
        self.miRNA_semantic_embedding = pd.read_csv(args.mirna_des_embedding, header=0).iloc[:, 1:].values

        # Drug特征
        self.drug_atom = pd.read_csv(args.drug_atom_feature, header=None).values
        self.drug_gin = pd.read_csv(args.drug_gin_features, header=None).values
        self.drug_maccs = pd.read_csv(args.drug_MACCS, header=None).values
        self.drug_sim = pd.read_csv(args.drug_similarity, header=0).iloc[:, 1:].values
        self.drug_semantic_embedding = pd.read_csv(args.drug_des_embedding, header=0).iloc[:, 1:].values

        # 构建miRNA相似度图
        self.miRNA_sim_edge_index = load_sim_edge_index(
            sim_file=args.mirna_similarity,
            sim_threshold=args.similarity_threshold
        )
        print(f"miRNA similarity graph: {self.miRNA_sim_edge_index.shape[1]} edges ")

        # 构建Drug相似度图
        self.drug_sim_edge_index = load_sim_edge_index(
            sim_file=args.drug_similarity,
            sim_threshold=args.similarity_threshold
        )
        print(f"Drug similarity graph:  {self.drug_sim_edge_index.shape[1]} edges ")


        # 使用GCN处理融合的4种特征（doc2vec, kmer, rnafm, similarity）
        self.miRNA_gcn_extractor = GCNFeatureExtractor(
            feature_dims=[args.doc2vec_dim, args.kmer_dim, args.rnafm_dim, args.miRNA_numbers],
            output_dim=args.pro_dim,
            dropout=args.fcDropout
        ).to(device)

        self.drug_gcn_extractor = GCNFeatureExtractor(
            feature_dims=[args.atom_dim, args.gin_dim, args.maccs_dim, args.drug_numbers],
            output_dim=args.pro_dim,
            dropout=args.fcDropout
        ).to(device)

        miRNA_total_dim = args.doc2vec_dim + args.kmer_dim + args.rnafm_dim + args.miRNA_numbers
        drug_total_dim = args.atom_dim + args.gin_dim + args.maccs_dim + args.drug_numbers
        print(f"miRNA GCN: 4 features fused ({miRNA_total_dim}d) → {args.pro_dim}d")
        print(f"Drug GCN:  4 features fused ({drug_total_dim}d) → {args.pro_dim}d")

        # BCQE-CA跨模态融合模块
        gcn_output_dim = args.pro_dim

        # BCQE-CA 投影层
        self.gcn_proj_m = nn.Linear(gcn_output_dim, args.embedding_dim)  # pro_dim(256) → embedding_dim(256)
        self.gcn_proj_d = nn.Linear(gcn_output_dim, args.embedding_dim)  # pro_dim(256) → embedding_dim(256)
        self.semantic_proj_m = nn.Linear(args.mirna_semantic_dim, args.embedding_dim)  # 768 → embedding_dim(256)
        self.semantic_proj_d = nn.Linear(args.drug_semantic_dim, args.embedding_dim)  # 768 → embedding_dim(256)

        self.bcqe_block_m = MultiModalCCABlock(
            dim=args.embedding_dim,
            num_heads=args.num_heads,
            mlp_ratio=4,
            dropout=args.fcDropout
        )
        self.bcqe_block_d = MultiModalCCABlock(
            dim=args.embedding_dim,
            num_heads=args.num_heads,
            mlp_ratio=4,
            dropout=args.fcDropout
        )

        self.bcqe_output_proj_m = nn.Linear(args.embedding_dim * 2, args.embedding_dim * 2)
        self.bcqe_output_proj_d = nn.Linear(args.embedding_dim * 2, args.embedding_dim * 2)

        # 可学习门控参数
        self.bcqe_fusion_gate_m = nn.Parameter(torch.zeros(1))
        self.bcqe_fusion_gate_d = nn.Parameter(torch.zeros(1))


        # 计算拼接后的总维度
        miRNA_concat_dim = args.doc2vec_dim + args.kmer_dim + args.rnafm_dim + args.miRNA_numbers
        drug_concat_dim = args.atom_dim + args.gin_dim + args.maccs_dim + args.drug_numbers
        
        # 投影到统一维度
        self.concat_proj_m = nn.Linear(miRNA_concat_dim, args.embedding_dim)
        self.concat_proj_d = nn.Linear(drug_concat_dim, args.embedding_dim)


        # 语义特征投影层
        self.pj_m_semantic = nn.Linear(args.mirna_semantic_dim, args.embedding_dim)
        self.pj_d_semantic = nn.Linear(args.drug_semantic_dim, args.embedding_dim)

        #异构图神经网络

        node_types = ['miRNA', 'drug']
        edge_types = [
            ('miRNA', 'interacts', 'drug'),
            ('drug', 'interacts_rev', 'miRNA')
        ]
        metadata = (node_types, edge_types)

        # 异构图输出维度512
        hetgnn_output_dim = 512
        
        self.hetgnn_concat = HeteroGNN(
            in_dims={'miRNA': args.embedding_dim, 'drug': args.embedding_dim},
            hidden_dim=args.hidden_dim,
            out_dim=hetgnn_output_dim,
            metadata=metadata,
            num_layers=args.num_layers,
            dropout=args.fcDropout
        )

        self.hetgnn_semantic = HeteroGNN(
            in_dims={'miRNA': args.embedding_dim, 'drug': args.embedding_dim},
            hidden_dim=args.hidden_dim,
            out_dim=hetgnn_output_dim,
            metadata=metadata,
            num_layers=args.num_layers,
            dropout=args.fcDropout
        )


        #预测层（SENet通道注意力融合）
        self.dropout = nn.Dropout(p=args.fcDropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        concat_dim = args.embedding_dim * 2 + hetgnn_output_dim * 2
        
        # SENet通道注意力模块（自适应特征重标定）
        self.se_block = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 8),
            nn.ReLU(),
            nn.Linear(concat_dim // 8, concat_dim),
            nn.Sigmoid()
        )
        
        # 残差预测层
        self.residual_predictor = ResidualPredictor(
            inSize=concat_dim,
            hiddenSize=512,
            outSize=args.outSize,
            dropout=args.fcDropout,
            num_layers=2
        )


        # ==================== 交互图缓存（每个fold重建）====================
        self.current_interaction_edge_index_dict = None  # 存储当前fold的交互图


        self.to(args.device)


    def build_graph(self, train_pairs, train_labels):
        device = self.args.device


        # 只使用正样本构建交互图
        interaction_edge_index = load_edge_ws_index(
            train_pairs,
            train_labels,
            num_mirna=self.args.miRNA_numbers
        ).to(device)

        # 正向边：miRNA->drug
        pos_edge = interaction_edge_index
        # 反向边：drug->miRNA
        rev_edge = torch.stack([pos_edge[1], pos_edge[0]], dim=0)

        self.current_interaction_edge_index_dict = {
            ('miRNA', 'interacts', 'drug'): pos_edge,
            ('drug', 'interacts_rev', 'miRNA'): rev_edge
        }

        num_edges = interaction_edge_index.shape[1]
        num_unique_edges = num_edges

        print(f"{num_unique_edges} unique edges")
        print(f"{num_edges} total edges (bidirectional)")


        return self.current_interaction_edge_index_dict

    def extract_gcn_features(self):
        device = self.args.device

        # miRNA特征通过相似度图传播（包含4种特征：doc2vec, kmer, rnafm, similarity）
        miRNA_features = [
            torch.from_numpy(self.miRNA_doc2vec).float().to(device),
            torch.from_numpy(self.miRNA_kmer).float().to(device),
            torch.from_numpy(self.miRNA_rnafm).float().to(device),
            torch.from_numpy(self.miRNA_sim).float().to(device)
        ]
        miRNA_gcn_fea = self.miRNA_gcn_extractor(
            miRNA_features,
            self.miRNA_sim_edge_index.to(device)
        )  # [701, 128]

        # Drug特征通过相似度图传播（包含4种特征：atom, gin, maccs, similarity）
        drug_features = [
            torch.from_numpy(self.drug_atom).float().to(device),
            torch.from_numpy(self.drug_gin).float().to(device),
            torch.from_numpy(self.drug_maccs).float().to(device),
            torch.from_numpy(self.drug_sim).float().to(device)
        ]
        drug_gcn_fea = self.drug_gcn_extractor(
            drug_features,
            self.drug_sim_edge_index.to(device)
        )  # [101, 128]

        return miRNA_gcn_fea, drug_gcn_fea

    def extract_semantic_features(self):

        device = self.args.device
        miRNA_semantic = torch.from_numpy(self.miRNA_semantic_embedding).float().to(device)
        drug_semantic = torch.from_numpy(self.drug_semantic_embedding).float().to(device)
        return miRNA_semantic, drug_semantic

    def bcqe_cross_modal_fusion(self, gcn_feat, semantic_feat, gcn_proj, semantic_proj, bcqe_block, output_proj, fusion_gate):

        # 投影到统一维度
        gcn_projected = self.dropout(gcn_proj(gcn_feat))
        semantic_projected = self.dropout(semantic_proj(semantic_feat))

        # 转换为序列格式用于Transformer
        gcn_seq = gcn_projected.unsqueeze(0)
        semantic_seq = semantic_projected.unsqueeze(0)

        # BCQE-CA Transformer融合
        fused_node_level = bcqe_block(gcn_seq, semantic_seq)
        fused_node_level = fused_node_level.squeeze(0)

        # 输出投影
        bcqe_output = output_proj(fused_node_level)

        # 可学习门控融合
        residual = torch.cat([gcn_projected, semantic_projected], dim=-1)
        gate = torch.sigmoid(fusion_gate)
        fused_final = gate * bcqe_output + (1 - gate) * residual

        return fused_final

    def extract_concat_features(self):

        device = self.args.device

        # miRNA特征拼接
        miRNA_features = torch.cat([
            torch.from_numpy(self.miRNA_doc2vec).float().to(device),
            torch.from_numpy(self.miRNA_kmer).float().to(device),
            torch.from_numpy(self.miRNA_rnafm).float().to(device),
            torch.from_numpy(self.miRNA_sim).float().to(device)
        ], dim=1)  # [N_miRNA, doc2vec_dim + kmer_dim + rnafm_dim + miRNA_numbers]
        
        miRNA_concat_fea = self.dropout(self.concat_proj_m(miRNA_features))  # [N_miRNA, embedding_dim]

        # Drug特征拼接
        drug_features = torch.cat([
            torch.from_numpy(self.drug_atom).float().to(device),
            torch.from_numpy(self.drug_gin).float().to(device),
            torch.from_numpy(self.drug_maccs).float().to(device),
            torch.from_numpy(self.drug_sim).float().to(device)
        ], dim=1)  # [N_drug, atom_dim + gin_dim + maccs_dim + drug_numbers]
        
        drug_concat_fea = self.dropout(self.concat_proj_d(drug_features))  # [N_drug, embedding_dim]

        return miRNA_concat_fea, drug_concat_fea

    def forward(self, batch_data, batch_labels, edge_index_dict=None):
        """前向传播"""
        device = self.args.device

        # 特征提取
        miRNA_gcn_fea, drug_gcn_fea = self.extract_gcn_features()
        miRNA_semantic_fea, drug_semantic_fea = self.extract_semantic_features()

        # BCQE-CA跨模态融合
        miRNA_bcqe_fea = self.bcqe_cross_modal_fusion(
            miRNA_gcn_fea, miRNA_semantic_fea,
            self.gcn_proj_m, self.semantic_proj_m,
            self.bcqe_block_m, self.bcqe_output_proj_m,
            self.bcqe_fusion_gate_m
        )

        drug_bcqe_fea = self.bcqe_cross_modal_fusion(
            drug_gcn_fea, drug_semantic_fea,
            self.gcn_proj_d, self.semantic_proj_d,
            self.bcqe_block_d, self.bcqe_output_proj_d,
            self.bcqe_fusion_gate_d
        )  # [101, 512]

        # 直接拼接特征
        miRNA_concat_fea, drug_concat_fea = self.extract_concat_features()

        # 语义特征投影
        miRNA_semantic_proj = self.dropout(self.pj_m_semantic(miRNA_semantic_fea))
        drug_semantic_proj = self.dropout(self.pj_d_semantic(drug_semantic_fea))

        # 异构图神经网络
        if edge_index_dict is None:
            if self.current_interaction_edge_index_dict is None:
                raise ValueError("❌ No interaction graph available!")
            edge_index_dict = self.current_interaction_edge_index_dict

        x_dict_concat = {'miRNA': miRNA_concat_fea, 'drug': drug_concat_fea}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            src_nodes, dst_nodes = edge_index[0], edge_index[1]

            if src_nodes.max() >= x_dict_concat[src_type].shape[0]:
                raise ValueError(
                    f"源节点索引{src_nodes.max()}超出特征维度{x_dict_concat[src_type].shape[0]}，节点类型：{src_type}")
            if dst_nodes.max() >= x_dict_concat[dst_type].shape[0]:
                raise ValueError(
                    f"目标节点索引{dst_nodes.max()}超出特征维度{x_dict_concat[dst_type].shape[0]}，节点类型：{dst_type}")


        hetgnn1_out = self.hetgnn_concat(x_dict_concat, edge_index_dict)

        x_dict_semantic = {'miRNA': miRNA_semantic_proj, 'drug': drug_semantic_proj}
        hetgnn2_out = self.hetgnn_semantic(x_dict_semantic, edge_index_dict)

        # 提取batch特征
        edgeData = batch_data.t()
        m_index = edgeData[0]  # miRNA 索引 [0, 700]
        d_index_raw = edgeData[1]  # Drug 全局索引 [701, 801]
        
        # 将 Drug 全局索引转换为局部索引 [0, 100]
        d_index = d_index_raw - self.args.miRNA_numbers

        # 验证索引范围
        if m_index.min() < 0 or m_index.max() >= self.args.miRNA_numbers:
            raise ValueError(
                f"miRNA index out of range: [{m_index.min()}, {m_index.max()}], "
                f"expected [0, {self.args.miRNA_numbers})"
            )
        if d_index.min() < 0 or d_index.max() >= self.args.drug_numbers:
            raise ValueError(
                f"Drug index out of range: [{d_index.min()}, {d_index.max()}], "
                f"expected [0, {self.args.drug_numbers})"
            )

        # 提取特征
        bcqe_m = torch.index_select(miRNA_bcqe_fea, 0, m_index)
        bcqe_d = torch.index_select(drug_bcqe_fea, 0, d_index)
        h1_m = torch.index_select(hetgnn1_out['miRNA'], 0, m_index)
        h1_d = torch.index_select(hetgnn1_out['drug'], 0, d_index)
        h2_m = torch.index_select(hetgnn2_out['miRNA'], 0, m_index)
        h2_d = torch.index_select(hetgnn2_out['drug'], 0, d_index)

        # 预测（SENet通道注意力 + 残差预测）
        # 拼接miRNA和Drug的所有特征
        final_miRNA_fea = torch.cat([bcqe_m, h1_m, h2_m], dim=1)
        final_drug_fea = torch.cat([bcqe_d, h1_d, h2_d], dim=1)
        
        # 元素级乘法（捕获miRNA-Drug交互）
        interaction_fea = final_miRNA_fea * final_drug_fea
        
        # SENet通道注意力（自适应重标定特征通道）
        attention_weights = self.se_block(interaction_fea)
        refined_fea = interaction_fea * attention_weights
        
        # 残差预测
        pre_part = self.residual_predictor(refined_fea)
        predictions = self.sigmoid(pre_part).squeeze(1)

        return predictions