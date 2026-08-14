import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import joblib
from sklearn.preprocessing import LabelEncoder # 需要引入
from train_mutation_transformer import MutationSetTransformer,PMA
import argparse

# --- 1. 配置 (必须与训练时完全一致) ---
CONFIG = {
    # --- 文件路径 ---
    # 'input_file': 'pan-wes-ici/pan_wes_ici_data_mutations_annotated_with_aachange.tsv', # 必须使用与训练时相同格式的输入文件
    # 'checkpoint_dir': './cbioportal_checkpoints',
    # 'output_file': './cbioportal_checkpoints/pan_wes_ici_patient_embeddings.npy',
    # 'output_pids': './cbioportal_checkpoints/pan_wes_ici_patient_ids.tsv',
    
    # --- 与训练时一致的数据和模型参数 ---
    'max_seq_len': 256,
    'aa_list': ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'STOP', 'UNK'],
    
    # --- 模型参数 (从训练脚本复制) ---
    
    'gene_embed_dim': 256,
    'mut_type_embed_dim': 128,
    'am_score_embed_dim': 64,
    'aa_embed_dim': 64,
    'rel_pos_embed_dim': 32,
    'embed_dim': 512,
    'n_layers': 6,
    'n_heads': 8,
    'ff_dim': 1024,
    'dropout': 0.3,
    'cl_proj_dim': 128,

}



# --- 3. 推理专用 Dataset ---
class InferenceDataset(Dataset):
    def __init__(self, samples, gene_le, mut_type_le,gene_ranks=None):
        self.samples = samples
        self.gene_le = gene_le
        self.mut_type_le = mut_type_le
        self.gene_ranks = gene_ranks
        # 定义与训练时一致的特殊 token ID
        self.pad_id = 0
        self.cls_token_gene_id = len(self.gene_le.classes_) + 2
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 截断逻辑
        L = len(sample['gene_ids'])
        max_len = CONFIG['max_seq_len'] - 1 
        indices = np.arange(L)[:max_len] if L > max_len else np.arange(L)
        current_len = len(indices)
        
        target_len = CONFIG['max_seq_len']
        
        # --- 初始化所有需要的输入 buffer ---
        padded_gene_ids = np.full(target_len, self.pad_id, dtype=np.int64)
        padded_mut_type_ids = np.full(target_len, self.pad_id, dtype=np.int64)
        padded_am_scores = np.zeros((target_len, 1), dtype=np.float32)
        padded_aa_wt = np.full(target_len, self.pad_id, dtype=np.int64)
        padded_aa_mt = np.full(target_len, self.pad_id, dtype=np.int64)
        padded_rel_pos = np.zeros((target_len, 1), dtype=np.float32)
        
        # --- 填充数据 (逻辑与训练脚本 _create_view 保持一致) ---
        # 1. 填充 [CLS] token
        padded_gene_ids[0] = self.cls_token_gene_id
        
        # 2. 填充真实突变数据
        padded_gene_ids[1 : current_len+1] = sample['gene_ids'][indices] + 1
        padded_mut_type_ids[1 : current_len+1] = sample['mut_type_ids'][indices] + 1
        padded_am_scores[1 : current_len+1] = sample['am_scores'][indices]
        padded_aa_wt[1 : current_len+1] = sample['aa_wt_ids'][indices]
        padded_aa_mt[1 : current_len+1] = sample['aa_mt_ids'][indices]
        padded_rel_pos[1 : current_len+1] = sample['rel_positions'][indices]
        
        # 3. 创建 Padding Mask (Transformer 用 True 表示 mask掉)
        padding_mask = np.ones(target_len, dtype=bool)
        padding_mask[:current_len+1] = False
        
        return {
            'pid': sample['pid'],
            'gene_ids': torch.tensor(padded_gene_ids),
            'mut_type_ids': torch.tensor(padded_mut_type_ids),
            'am_scores': torch.tensor(padded_am_scores),
            'aa_wt_ids': torch.tensor(padded_aa_wt),
            'aa_mt_ids': torch.tensor(padded_aa_mt),
            'rel_positions': torch.tensor(padded_rel_pos),
            'padding_mask': torch.tensor(padding_mask),
        }

# --- 4. 辅助函数 ---
def safe_transform(encoder, items, unknown_value=None):
    """
    将 item 列表转换为 ID。如果 item 未在 encoder 中，则使用 unknown_value。
    """
    if unknown_value is None:
        # 默认使用 encoder 中的 '<OTHER>' 或最后一个类别作为未知值
        if '<OTHER>' in encoder.classes_:
            unknown_value = encoder.transform(['<OTHER>'])[0]
        else:
            # 备用策略，可能不理想
            print("Warning: '<OTHER>' token not found. Unknown items will be mapped to the last class index.")
            unknown_value = len(encoder.classes_) - 1
            
    classes = set(encoder.classes_)
    return np.array([encoder.transform([item])[0] if item in classes else unknown_value for item in items])


def parse_args():
    parser = argparse.ArgumentParser(description="Extract tumor sample embeddings")

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入 mutation TSV 文件"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="模型和 encoder 所在目录"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出 embedding 的 .npy 文件"
    )

    parser.add_argument(
        "--output_pids",
        type=str,
        required=True,
        help="输出 patient id 的 .tsv 文件"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="推理 batch size"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="DataLoader workers"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备，例如 npu:0 / cpu / cuda:0"
    )

    return parser.parse_args()

# --- 5. 主流程 ---
def main():
    args = parse_args()

    if args.device is None:
        device = 'npu' if torch.npu.is_available() else 'cpu'
    else:
        device = args.device

    print(f"--- Starting Extraction on {device} ---")
    
    # 1. 加载编码器
    print("Loading encoders...")
    gene_le_path = os.path.join(
        args.checkpoint_dir,
        'gene_encoder.pkl'
    )

    mut_type_le_path = os.path.join(
        args.checkpoint_dir,
        'mut_type_encoder.pkl'
    )
    
    if not os.path.exists(gene_le_path) or not os.path.exists(mut_type_le_path):
        raise FileNotFoundError("Encoders not found! Run training first.")
        
    gene_le = joblib.load(gene_le_path)
    mut_type_le = joblib.load(mut_type_le_path)
    
    gene_vocab_size = len(gene_le.classes_)
    mut_type_vocab_size = len(mut_type_le.classes_)
    print(f"Gene Vocab: {gene_vocab_size}, Mutation Type Vocab: {mut_type_vocab_size}")

    # 2. 加载并处理数据 (与训练时逻辑保持一致)
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file, sep='\t')
    
    # 清理和预处理
    required_cols = ['Hugo_Symbol', 'SBS96', 'am_pathogenicity', 'Tumor_Sample_Barcode', 'AAchange', 'Relative_Position']
    df = df.dropna(subset=required_cols).copy()
    
    # 创建氨基酸映射
    aa_to_id = {aa: i for i, aa in enumerate(CONFIG['aa_list'])}
    unk_aa_id = aa_to_id['UNK']
    def parse_aa_change(x):
        try:
            if '>' in str(x):
                parts = x.split('>')
                return aa_to_id.get(parts[0], unk_aa_id), aa_to_id.get(parts[1], unk_aa_id)
        except: pass
        return unk_aa_id, unk_aa_id

    print("Encoding features...")
    df['gene_id'] = safe_transform(gene_le, df['Hugo_Symbol'].values)
    df['mut_type_id'] = safe_transform(mut_type_le, df['SBS96'].values)
    aa_features = df['AAchange'].apply(parse_aa_change)
    df['aa_wt_id'] = [x[0] for x in aa_features]
    df['aa_mt_id'] = [x[1] for x in aa_features]
    df['rel_pos'] = df['Relative_Position'].fillna(0).astype(float)
    
    # 按患者聚合
    print("Grouping and sorting samples...")
    samples = []
    df['Tumor_Sample_Barcode'] = df['Tumor_Sample_Barcode'].astype(str)
    
    # 使用 groupby, apply 比循环更快
    grouped = df.groupby('Tumor_Sample_Barcode')
    for pid, group in tqdm(grouped, desc="Processing patients"):
        # 排序 (am_pathogenicity 降序, 与训练一致)
        group = group.sort_values('am_pathogenicity', ascending=False)
        
        samples.append({
            'pid': pid,
            'gene_ids': group['gene_id'].values,
            'mut_type_ids': group['mut_type_id'].values,
            'am_scores': group['am_pathogenicity'].values.reshape(-1, 1).astype(np.float32),
            'aa_wt_ids': group['aa_wt_id'].values,
            'aa_mt_ids': group['aa_mt_id'].values,
            'rel_positions': group['rel_pos'].values.reshape(-1, 1).astype(np.float32)
        })
        
    dataset = InferenceDataset(samples, gene_le, mut_type_le)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Prepared {len(samples)} patients for extraction.")

    # 3. 加载模型
    print("Loading Best Model...")
    model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
        
    model = MutationSetTransformer(CONFIG, gene_vocab_size, mut_type_vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 关键！关闭 Dropout

    # 4. 提取特征
    print("Extracting features...")
    all_embeddings = []
    all_pids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # 将所有需要的输入移动到设备
            gene_ids = batch['gene_ids'].to(device)
            mut_type_ids = batch['mut_type_ids'].to(device)
            am_scores = batch['am_scores'].to(device)
            aa_wt_ids = batch['aa_wt_ids'].to(device)
            aa_mt_ids = batch['aa_mt_ids'].to(device)
            rel_positions = batch['rel_positions'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            pids = batch['pid']
            
            # Forward pass, 获取 patient_representation
            _, patient_representation, _ = model(gene_ids, mut_type_ids, am_scores, aa_wt_ids, aa_mt_ids, rel_positions, padding_mask)
            
            all_embeddings.append(patient_representation.cpu().numpy())
            all_pids.extend(pids)
            
    # 5. 保存结果
    final_embs = np.vstack(all_embeddings)
    print(f"\nExtraction Complete. Shape: {final_embs.shape}")
    
    np.save(args.output_file, final_embs)
    print(f"Saved embeddings to {args.output_file}")
    
    pid_df = pd.DataFrame({'Tumor_Sample_Barcode': all_pids})
    pid_df.to_csv(args.output_pids, sep='\t', index=False)
    print(f"Saved patient IDs to {args.output_pids}")



if __name__ == "__main__":
    main()