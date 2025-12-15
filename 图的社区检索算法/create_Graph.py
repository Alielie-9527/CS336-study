import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import os

# 加载多语言句子嵌入模型,用于生成句子的嵌入向量，方便后续的相似度检查
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  

DATA_PATH = './data/harrypotterkgdata.csv'

# 第一部分构建Graph类
class Knowledgebase:
    def __init__(self,data_path=DATA_PATH):
        self.graph = defaultdict(list)
        self.nodes = set()
        self.embeddings = {}

        df = pd.read_csv(data_path,header=None,names=['head','tail','realtion'])
        df.dropna(inplace=True)
        for _,row in df.iterrows():
            head = row['head'].strip()
            tail = row['tail'].strip()
            relation = row['realtion'].strip()

            # 图的；邻接表表示
            self.graph[head].append((relation, tail))
            self.nodes.add(head)
            self.nodes.add(tail)
        
        # 生成节点的向量表示
        node_list = list(self.nodes)
        embeddings = model.encode(node_list, show_progress_bar=True) 

        for node, emb in zip(node_list, embeddings):
            self.embeddings[node] = emb
    
    def get_neighbors(self, node):
        return self.graph.get(node, [])

    def get_embedding(self, node):
        return self.embeddings.get(node, None)

    def get_all_nodes(self):
        return list(self.nodes)
    
# # --- 测试部分 ---
# if __name__ == "__main__":
#     kb = Knowledgebase(DATA_PATH)
#     print(f"\n✅ 中文图谱构建成功！")
#     print(f"总节点数: {len(kb.nodes)}")
    
#     test_node = "乔治·韦斯莱"
#     if test_node in kb.nodes:
#         print(f"\n[{test_node}] 的关系网示例:")
#         # 打印前5个邻居
#         for neighbor, relation in kb.get_neighbors(test_node)[:5]:
#             print(f"  --[{relation}]--> {neighbor}")


# LPA算法实现社区发现
kb = Knowledgebase(DATA_PATH)
graph = kb.graph
all_nodes = kb.get_all_nodes()

# 初始化标签，每个节点的标签初始为其自身
def init_labels(nodes):
    labels= {node: node for node in nodes}
    return labels

labels = init_labels(all_nodes)

# LPA算法核心步骤： 投票
