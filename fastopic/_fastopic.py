import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

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
            if cell_embeddings is not None:
                kmeans = KMeans(n_clusters=self.num_topics, random_state=42, n_init=10)
                kmeans.fit(cell_embeddings)
                topic_embeddings = F.normalize(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
            else:
                topic_embeddings = F.normalize(nn.init.trunc_normal_(torch.empty((self.num_topics, embed_size))))
            topic_weights = (torch.ones(self.num_topics) / self.num_topics).unsqueeze(1)

        self.topic_embeddings = nn.Parameter(topic_embeddings)
        self.topic_weights = nn.Parameter(topic_weights)

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

        loss_ETP = loss_DT + loss_TW

        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        
        recon_logit = torch.matmul(theta, beta)
        train_bow_prob = train_bow / train_bow.sum(dim=-1, keepdim=True)
        recon_prob = F.softmax(recon_logit, dim=-1)
        loss_DSR = -(train_bow_prob * recon_prob.log()).sum(dim=-1).mean()
        loss = loss_DSR + loss_ETP 

        rst_dict = {
            'loss': loss,
            'loss_ETP': loss_ETP,
            'loss_DSR': loss_DSR,
            'loss_DT': loss_DT,
            'loss_TW': loss_TW,
        }

        return rst_dict
