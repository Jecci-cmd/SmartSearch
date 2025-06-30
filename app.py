# 提供一个/search接口，根据用户输入的query进行语义搜索，返回最相关的论文

from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import torch

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载模型和FAISS索引
model_path = r"bge-base-en-v1.5"
output_dir = r"."  # FAISS索引和记录存储的目录
index_file = os.path.join(output_dir, "crossref_bge_base.faiss") # faiss索引文件
records_file = os.path.join(output_dir, "crossref_bge_base_records.json") # 存储论文的原始信息，DOI为键
id_map_file = os.path.join(output_dir, "crossref_bge_base_id_map.json") # 建立了FAISS索引的向量ID和DOI之间的映射

model = SentenceTransformer(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
index = faiss.read_index(index_file) # 读取FAISS索引

with open(records_file, 'r', encoding='utf-8') as f:
    records_store = json.load(f)
with open(id_map_file, 'r', encoding='utf-8') as f:
    id_map = json.load(f)

@app.route('/search', methods=['POST']) # 前端POST一个query和可选的year,用于语义搜索，拿回top10的结果
def search():
    data = request.get_json()
    query = data.get('query', '')
    year_filter = data.get('year', '')

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    # 编码查询
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding], dtype=np.float32)

    # 在FAISS中搜索
    k = 10  # 返回前10个结果
    distances, indices = index.search(query_embedding, k)

    # 获取结果
    results = []
    for idx in indices[0]:
        vector_id = id_map.get(str(idx), '')
        if not vector_id:
            continue
        doi = vector_id.replace('title_', '').replace('abs_', '')
        record = records_store.get(doi)
        if record:
            if year_filter and str(record.get('file_idx', '')) != year_filter:
                continue
            results.append({
                'title': record.get('title', ''),
                'abstract': record.get('abstract', ''),
                'doi': doi
            })

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)