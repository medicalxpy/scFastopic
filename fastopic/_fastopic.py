import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import pickle
import warnings

from ._ETP import ETP
from ._model_utils import pairwise_euclidean_distance



class fastopic(nn.Module):
    def __init__(self,
                 num_topics: int,
                 theta_temp: float=1.0,
                 DT_alpha: float=3.0,
                 TW_alpha: float=2.0
                ):
        super().__init__()

        self.num_topics = num_topics
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp

        self.epsilon = 1e-12

    def init(self,
             vocab_size: int,
             embed_size: int,
             _fitted: bool = False,
             pre_vocab: list=None,
             vocab: list=None,
             cell_embeddings: np.ndarray=None
            ):

        if _fitted:
            topic_embeddings = self.topic_embeddings.data
            assert topic_embeddings.shape == (self.num_topics, embed_size)
            topic_weights = self.topic_weights.data
            del self.topic_weights
        else:
            topic_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty((self.num_topics, embed_size))))
            topic_weights = (torch.ones(self.num_topics) / self.num_topics).unsqueeze(1)

        self.topic_embeddings = nn.Parameter(topic_embeddings)
        self.topic_weights = nn.Parameter(topic_weights)

        # 初始化word embeddings - 使用随机初始化
        word_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty(vocab_size, embed_size)))
        
        if _fitted:
            pre_word_embeddings = self.word_embeddings.data
            word_weights = torch.zeros(vocab_size, 1)
            pre_norm_word_weights = F.softmax(self.word_weights.data, dim=0)
            del self.word_embeddings
            del self.word_weights

            for i, word in enumerate(vocab):
                if word in pre_vocab:
                    pre_word_idx = pre_vocab.index(word)
                    word_embeddings[i] = pre_word_embeddings[pre_word_idx]
                    word_weights[i] = pre_norm_word_weights[pre_word_idx]

            left_avg = (1.0 - word_weights.sum()) / word_weights.nonzero().size(0)
            word_weights[word_weights == 0] = left_avg

            word_weights = torch.log(word_weights)
            word_weights = word_weights - word_weights.mean()

        else:
            word_weights = (torch.ones(vocab_size) / vocab_size).unsqueeze(1)

        self.word_embeddings = nn.Parameter(word_embeddings)
        self.word_weights = nn.Parameter(word_weights)
        
        # 保存vocab_size用于计算权重
        self.vocab_size = vocab_size
        
        # 保存词汇表用于GenePT对齐
        self._vocab = vocab

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

    def get_transp_DT(self, doc_embeddings):
        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

        return transp.detach().cpu().numpy()

    # only for testing
    def get_beta(self):
        with torch.no_grad():
            _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
            # use transport plan as beta
            beta = transp_TW * transp_TW.shape[0]

            return beta

    # only for testing
    def get_theta(self,
            doc_embeddings,
            train_doc_embeddings
        ):
        with torch.no_grad():
            topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
            dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
            train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)

            exp_dist = torch.exp(-dist / self.theta_temp)
            exp_train_dist = torch.exp(-train_dist / self.theta_temp)

            theta = exp_dist / (exp_train_dist.sum(0))
            theta = theta / theta.sum(1, keepdim=True)

            return theta

    def forward(self, train_bow, doc_embeddings):
        loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        # 添加权重的损失计算
        num_cells = doc_embeddings.shape[0]
        dt_weight = (self.num_topics / num_cells) ** 0.5
        tw_weight = (self.vocab_size / self.num_topics) ** 0.5
        loss_ETP = dt_weight * loss_DT + tw_weight * loss_TW
        
        # # 自动尺度平衡 - 计算距离矩阵来归一化损失
        # from ._model_utils import pairwise_euclidean_distance
        # M_DT = pairwise_euclidean_distance(doc_embeddings, self.topic_embeddings)
        # M_TW = pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        
        # # 按距离尺度归一化损失，使DT和TW损失在相似的量级
        # loss_DT_normalized = loss_DT / M_DT.mean()
        # loss_TW_normalized = loss_TW / M_TW.mean()
        # loss_ETP = loss_DT_normalized + loss_TW_normalized
        
        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        
        recon = torch.matmul(theta, beta)
        loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()
        
        # 添加GenePT对齐损失
        loss_genept_alignment = self._compute_genept_alignment_loss()
        
        loss = loss_DSR + 1e-2*loss_ETP + 1e-4*loss_genept_alignment 

        rst_dict = {
            'loss': loss,
            'loss_ETP': loss_ETP,
            'loss_DSR': loss_DSR,
            'loss_DT': loss_DT,
            'loss_TW': loss_TW,
            'loss_genept_alignment': loss_genept_alignment,
        }

        return rst_dict
    
    def _compute_genept_alignment_loss(self):
        """
        计算GenePT对齐损失，约束学到的gene embedding与预训练GenePT embedding对齐
        """
        # 如果没有词汇表信息，返回0损失
        if not hasattr(self, '_vocab') or self._vocab is None:
            return torch.tensor(0.0, device=self.word_embeddings.device)
        
        # 延迟加载GenePT embeddings
        if not hasattr(self, '_genept_embeddings') or self._genept_embeddings is None:
            self._load_genept_embeddings()
        
        # 如果加载失败或没有对齐的基因，返回0损失
        if self._genept_embeddings is None or self._gene_alignment_mask is None:
            return torch.tensor(0.0, device=self.word_embeddings.device)
        
        # 计算对齐的基因的embedding距离
        # 只对能匹配到GenePT的基因计算损失
        aligned_word_embeddings = self.word_embeddings[self._gene_alignment_mask]
        aligned_genept_embeddings = self._genept_embeddings.to(self.word_embeddings.device)
        
        # 使用余弦相似度损失 (1 - cosine_similarity)
        # 或者使用L2距离
        cosine_sim = F.cosine_similarity(aligned_word_embeddings, aligned_genept_embeddings, dim=1)
        alignment_loss = (1.0 - cosine_sim).mean()
        
        # 调节损失权重
        genept_alpha = 0.1  # 可调节的权重参数
        
        return genept_alpha * alignment_loss
    
    def _load_genept_embeddings(self):
        """
        加载GenePT基因嵌入并创建对齐掩码
        """
        try:
            genept_path = '/root/autodl-tmp/scFastopic/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle'
            
            with open(genept_path, 'rb') as f:
                genept_dict = pickle.load(f)
            
            # 创建对齐列表和掩码
            aligned_embeddings = []
            alignment_mask = []
            
            for i, gene_name in enumerate(self._vocab):
                if gene_name in genept_dict:
                    genept_emb = torch.tensor(genept_dict[gene_name], dtype=torch.float32)
                    aligned_embeddings.append(genept_emb)
                    alignment_mask.append(i)
            
            if len(aligned_embeddings) > 0:
                self._genept_embeddings = torch.stack(aligned_embeddings)
                self._gene_alignment_mask = torch.tensor(alignment_mask, dtype=torch.long)
                print(f"✅ GenePT对齐: {len(aligned_embeddings)}/{len(self._vocab)} 基因匹配")
            else:
                self._genept_embeddings = None
                self._gene_alignment_mask = None
                print("⚠️ 没有基因能与GenePT对齐")
                
        except Exception as e:
            print(f"❌ 加载GenePT嵌入失败: {e}")
            self._genept_embeddings = None
            self._gene_alignment_mask = None
