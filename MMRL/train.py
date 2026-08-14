import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import joblib
import pickle
import torch_npu
#os.environ["npu_VISIBLE_DEVICES"] = "1"

# --- 1. 全局配置 (Unchanged) ---
CONFIG = {
    # --- 数据路径 ---
    'input_file': 'final_filtered_cleaned_with_aachange.tsv',
    
    'cgc_file': 'Compendium_Cancer_Genes.tsv',
    'fp_driver_file': 'False_Positive_Drivers.txt',
    
    'output_dir': './cbioportal_checkpoints',

     # --- 新增氨基酸词表 (20种基本氨基酸 + PAD + UNK + STOP) ---
    'aa_list': ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'STOP', 'UNK'],


    # --- 数据参数 ---
    'max_seq_len': 256,
    'top_k_genes': 8000,
    'batch_size': 256,
    
    # --- Sample Filtering Parameters ---
    'min_mutations': 10,      # Minimum number of total mutations required for a sample
    'min_unique_genes': 3,   # Minimum number of unique genes required for a sample

    # --- 模型参数 ---

    'gene_embed_dim': 256,
    
    'mut_type_embed_dim': 128,
    
    'am_score_embed_dim': 64,
    # --- 新增嵌入维度 ---
    'aa_embed_dim': 64,        # 氨基酸类型嵌入
    'rel_pos_embed_dim': 32,   # 相对位置投影维度

    'embed_dim': 512,
    'n_layers': 6,
    'n_heads': 8,
    'ff_dim': 1024,
    'dropout': 0.3,

    # --- 对比学习参数 ---
    'lambda_mlm': 1.0,
    'lambda_cl': 0.1,
    'cl_temp': 0.2,
    'cl_proj_dim': 128,

    # --- 训练与验证参数 ---
    'epochs': 500,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'patience': 30,
    'seed': 42,
    'num_workers': 16,

}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
if torch.npu.is_available():
    torch.npu.manual_seed_all(CONFIG['seed'])

# --- 2. 数据集定义 (Unchanged) ---
class MutationSetDataset(Dataset):
    def __init__(self, samples, gene_le,  mut_type_le, gene_ranks, perform_masking=True, perform_aug=False,fixed_masking=False,seed=42):
        self.samples = samples
        self.gene_ranks = gene_ranks # 基因的频数排名
        self.gene_le,  self.mut_type_le = gene_le,  mut_type_le
        self.perform_masking, self.perform_aug = perform_masking, perform_aug
        self.fixed_masking = fixed_masking
        self.seed = seed
        self.pad_id, self.gene_mask_id, self.cls_token_gene_id = 0, len(self.gene_le.classes_)+1, len(self.gene_le.classes_) + 2

    def __len__(self):
        return len(self.samples)

    def _create_view(self, gene_ids_raw,  mut_type_ids_raw, am_scores_raw, am_class_ids_raw,aa_wt_raw, aa_mt_raw, rel_pos_raw,sample_idx=None,view_id=0):
        if self.perform_aug:
            benign_indices = np.where(am_class_ids_raw == 0)[0]
            if len(benign_indices) > 0:
                drop_ratio = np.random.uniform(0.1, 0.3)
                num_to_drop = int(len(benign_indices) * drop_ratio)
                drop_indices = np.random.choice(benign_indices, num_to_drop, replace=False)
                keep_mask = np.ones(len(gene_ids_raw), dtype=bool)
                keep_mask[drop_indices] = False
                gene_ids_raw,  mut_type_ids_raw, am_scores_raw, am_class_ids_raw,aa_wt_raw, aa_mt_raw, rel_pos_raw = \
                    gene_ids_raw[keep_mask],  mut_type_ids_raw[keep_mask], \
                    am_scores_raw[keep_mask], am_class_ids_raw[keep_mask], \
                    aa_wt_raw[keep_mask], aa_mt_raw[keep_mask], rel_pos_raw[keep_mask]

        L = len(gene_ids_raw)
        max_len = CONFIG['max_seq_len'] - 1
        indices = np.arange(L)[:max_len] if L > max_len else np.arange(L)
        if self.fixed_masking:
            rng = np.random.RandomState(self.seed + sample_idx * 1009 + view_id)
            rng.shuffle(indices)
        else:
            rng = np.random
            rng.shuffle(indices)
        #np.random.shuffle(indices) 
        current_len = len(indices)
        gene_ids,  mut_type_ids, am_scores = \
            gene_ids_raw[indices],  mut_type_ids_raw[indices], am_scores_raw[indices]

        labels_gene = np.full(CONFIG['max_seq_len'], -100, dtype=np.int64)
        input_gene_ids = gene_ids+1 # +1 因为 0 是 PAD

        if self.perform_masking:
            masked_any = False
            # 实现长尾加权掩码
            for i in range(current_len):
                raw_gid = gene_ids[i]  # 0-based gene id
                rank = self.gene_ranks.get(raw_gid, 9999)
                
                # 分层掩码概率
                if rank <= 500: mask_prob = 0.15
                elif rank <= 3000: mask_prob = 0.25
                else: mask_prob = 0.40
                
                if rng.rand() < mask_prob:
                    labels_gene[i+1] = raw_gid + 1   # label 仍然是 1 ~ N，0 是 PAD
                    input_gene_ids[i] = self.gene_mask_id # 替换为 MASK ID
                    masked_any = True

        if self.perform_masking and current_len > 0 and not masked_any:
            j = rng.randint(current_len)
            raw_gid = gene_ids[j]
            labels_gene[j + 1] = raw_gid + 1
            input_gene_ids[j] = self.gene_mask_id
        
        aa_wt = aa_wt_raw[indices]
        aa_mt = aa_mt_raw[indices]
        rel_pos = rel_pos_raw[indices]
        target_len = CONFIG['max_seq_len']
        padded_aa_wt = np.full(target_len, 0, dtype=np.int64) # 0 is PAD
        padded_aa_mt = np.full(target_len, 0, dtype=np.int64)
        padded_rel_pos = np.zeros((target_len, 1), dtype=np.float32)
        padded_gene_ids,  padded_mut_type_ids = \
            (np.full(target_len, self.pad_id, dtype=np.int64) for _ in range(2))
        padded_am_scores = np.zeros((target_len, 1), dtype=np.float32)

        padded_gene_ids[1:current_len+1] = input_gene_ids
        
        padded_mut_type_ids[1:current_len+1] = mut_type_ids + 1
        
        padded_am_scores[1:current_len+1] = am_scores
        padded_gene_ids[0] = self.cls_token_gene_id
        padded_aa_wt[1:current_len+1] = aa_wt
        padded_aa_mt[1:current_len+1] = aa_mt
        padded_rel_pos[1:current_len+1] = rel_pos

        padding_mask = np.ones(target_len, dtype=bool)
        padding_mask[:current_len+1] = False

        return {'gene_ids': torch.tensor(padded_gene_ids), 
                'mut_type_ids': torch.tensor(padded_mut_type_ids), 
                'am_scores': torch.tensor(padded_am_scores), 
                'aa_wt_ids': torch.tensor(padded_aa_wt),
                'aa_mt_ids': torch.tensor(padded_aa_mt),
                'rel_positions': torch.tensor(padded_rel_pos),
                'labels_gene': torch.tensor(labels_gene),
                'padding_mask': torch.tensor(padding_mask)}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = {'pid': sample['pid']}
        
        raw_features = (sample['gene_ids'],  sample['mut_type_ids'],
                        sample['am_scores'], sample['am_class_ids'],
                        sample['aa_wt_ids'], 
                        sample['aa_mt_ids'], sample['rel_positions'])
        if self.perform_aug:
            view1 = self._create_view(*raw_features, sample_idx=idx, view_id=1)
            view2 = self._create_view(*raw_features, sample_idx=idx, view_id=2)
            item.update({f"{k}_1": v for k, v in view1.items()})
            item.update({f"{k}_2": v for k, v in view2.items()})
        else:
            item.update(self._create_view(*raw_features, sample_idx=idx, view_id=0))
        return item

# --- 3. 数据预处理函数 (Unchanged) ---
def prepare_data(config):
    print("--- Starting Data Preparation ---")
    df = pd.read_csv(config['input_file'], sep='\t')
    required_cols = ['Hugo_Symbol', 'SBS96', 'am_pathogenicity', 'Tumor_Sample_Barcode',
                     'Variant_Classification', 'am_class','AAchange','Relative_Position']
    df = df.dropna(subset=required_cols)
    df = df.copy()

    print("1. Creating Hybrid Gene Vocabulary...")
    df_cgc = pd.read_csv(config['cgc_file'], sep='\t')
    cgc_genes = set(df_cgc['SYMBOL'].unique())
    with open(config['fp_driver_file'], 'r') as f: fp_genes = set([line.strip() for line in f])
    gene_counts = df[~df['Hugo_Symbol'].isin(fp_genes)].groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique().sort_values(ascending=False)
    top_k_genes = set(gene_counts.head(config['top_k_genes']).index)
    core_gene_vocab = sorted(list(cgc_genes.union(top_k_genes))) + ['<OTHER>']
    gene_le = LabelEncoder().fit(core_gene_vocab)
    gene_ranks = {gene_le.transform([g])[0]: i for i, g in enumerate(gene_counts.head(config['top_k_genes']).index)}
    other_token_id = gene_le.transform(['<OTHER>'])[0]
    df['gene_id'] = df['Hugo_Symbol'].apply(lambda x: gene_le.transform([x])[0] if x in gene_le.classes_ else other_token_id)

    print("2. Creating Feature Encoders...")
    mut_type_le = LabelEncoder().fit(df['SBS96'])
    df['mut_type_id'] = mut_type_le.transform(df['SBS96'])
    
    
    df['am_class_id'] = (df['am_class'] != 'benign').astype(int)

    print("3. Sorting mutations by pathogenicity score...")
    # 确保 ID 列是字符串，防止排序时 str 和 int 比较报错
    df['Tumor_Sample_Barcode'] = df['Tumor_Sample_Barcode'].astype(str) 
    df = df.sort_values(['Tumor_Sample_Barcode', 'am_pathogenicity'], ascending=[True, False])

    # 建立氨基酸映射表
    aa_to_id = {aa: i for i, aa in enumerate(config['aa_list'])}
    unk_id = aa_to_id['UNK']

    def parse_aa_change(x):
        try:
            if '>' in str(x):
                parts = x.split('>')
                return aa_to_id.get(parts[0], unk_id), aa_to_id.get(parts[1], unk_id)
        except:
            pass
        return unk_id, unk_id

    print("Parsing AA Changes and Positions...")
    # 提取 WT 和 MT
    aa_features = df['AAchange'].apply(parse_aa_change)
    df['aa_wt_id'] = [x[0] for x in aa_features]
    df['aa_mt_id'] = [x[1] for x in aa_features]
    
    # 确保相对位置在 0-1 之间
    df['rel_pos'] = df['Relative_Position'].fillna(0).astype(float)

    print("4. Grouping Samples...")
    
    pids = df['Tumor_Sample_Barcode'].values
    gene_ids = df['gene_id'].values
    mut_type_ids = df['mut_type_id'].values
    am_scores = df['am_pathogenicity'].values.reshape(-1, 1)
    am_class_ids = df['am_class_id'].values
    aa_wt_ids = df['aa_wt_id'].values
    aa_mt_ids = df['aa_mt_id'].values
    rel_positions = df['rel_pos'].values.reshape(-1, 1)

    # 找到每个样本边界的索引
    _, index = np.unique(pids, return_index=True)
    index = np.sort(index)
    split_indices = index[1:]

    # 使用 np.split 快速切割
    pids_split = np.split(pids, split_indices)
    genes_split = np.split(gene_ids, split_indices)
    muts_split = np.split(mut_type_ids, split_indices)
    ams_split = np.split(am_scores, split_indices)
    classes_split = np.split(am_class_ids, split_indices)
    wt_split = np.split(aa_wt_ids, split_indices)
    mt_split = np.split(aa_mt_ids, split_indices)
    pos_split = np.split(rel_positions, split_indices)

    all_samples = []
    for i in range(len(pids_split)):
        all_samples.append({
            'pid': pids_split[i][0],
            'gene_ids': genes_split[i],
            'mut_type_ids': muts_split[i],
            'am_scores': ams_split[i],
            'am_class_ids': classes_split[i],
            'aa_wt_ids': wt_split[i],
            'aa_mt_ids': mt_split[i],
            'rel_positions': pos_split[i]
        })
    
    print(f"5. Filtering samples... Initial count: {len(all_samples)}")
    initial_count = len(all_samples)
    all_samples_filtered = [
        s for s in all_samples
        if len(s['gene_ids']) >= config['min_mutations'] and len(np.unique(s['gene_ids'])) >= config['min_unique_genes']
    ]
    print(f"   Filtered out {initial_count - len(all_samples_filtered)} samples.")
    print(f"   Final sample count: {len(all_samples_filtered)}")

    
    train_samples, val_samples = train_test_split(all_samples_filtered, test_size=0.15, random_state=config['seed'])
    print(f"Data Split: {len(train_samples)} Train, {len(val_samples)} Val")
    return train_samples, val_samples, gene_le, mut_type_le, gene_ranks

# --- 4. 模型定义 (Unchanged) ---
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, dropout=0.1):
        super().__init__()
        self.num_seeds, self.seed_vectors = num_seeds, nn.Parameter(torch.randn(1, num_seeds, dim))
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm1, self.norm2, self.dropout = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.Dropout(dropout)
    def forward(self, x, mask=None):
        queries = self.seed_vectors.expand(x.size(0), -1, -1)
        attn_output, _ = self.attention(queries, x, x, key_padding_mask=mask)
        queries = self.norm1(queries + self.dropout(attn_output))
        ffn_output = self.ffn(queries)
        return self.norm2(queries + self.dropout(ffn_output))

class MutationSetTransformer(nn.Module):
    def __init__(self, config, gene_vocab_size,  mut_type_vocab_size):
        super().__init__()

        aa_vocab_size = len(config['aa_list'])
        self.aa_embed = nn.Embedding(aa_vocab_size, config['aa_embed_dim'], padding_idx=0)
        
        # 新增：相对位置线性投影
        self.rel_pos_proj = nn.Linear(1, config['rel_pos_embed_dim'])

        self.config = config
        dim = config['embed_dim']

        self.gene_embed = nn.Embedding(
            gene_vocab_size + 3,
            config['gene_embed_dim'],
            padding_idx=0
        )
        
        
        self.mut_type_embed = nn.Embedding(mut_type_vocab_size + 1, config['mut_type_embed_dim'], padding_idx=0)
        
        self.am_score_proj = nn.Linear(1, config['am_score_embed_dim'])
        
        fusion_input_dim = (
            config['gene_embed_dim'] + 
            config['mut_type_embed_dim'] + 
            config['am_score_embed_dim'] + 
            config['aa_embed_dim'] * 2 + # WT 和 MT 分开
            config['rel_pos_embed_dim']
        )
        self.fusion = nn.Sequential(nn.Linear(fusion_input_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(config['dropout']))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=config['n_heads'], dim_feedforward=config['ff_dim'], dropout=config['dropout'], activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['n_layers'])
        self.pooling = PMA(dim=dim, num_heads=config['n_heads'], num_seeds=8, dropout=config['dropout'])
        
        self.gene_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, gene_vocab_size + 1))
        self.cl_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, config['cl_proj_dim']))
        
    def forward(self, gene_ids,  mut_type_ids, am_scores,aa_wt_ids, aa_mt_ids, rel_positions, padding_mask):
        e_gene = self.gene_embed(gene_ids)
        
        e_mut_type = self.mut_type_embed(mut_type_ids)
        
        e_am_score = self.am_score_proj(am_scores)

        e_aa_wt = self.aa_embed(aa_wt_ids)
        e_aa_mt = self.aa_embed(aa_mt_ids)
        e_rel_pos = self.rel_pos_proj(rel_positions)
        
        concat = torch.cat([e_gene,  e_mut_type,  e_am_score, e_aa_wt, e_aa_mt, e_rel_pos], dim=-1)
        x = self.fusion(concat)
        encoder_output = self.encoder(x, src_key_padding_mask=padding_mask)
        
        logits_gene = self.gene_head(encoder_output)

        patient_representation = self.pooling(encoder_output, mask=padding_mask) # (B, 8, dim)
        # 展平或取平均作为最终表征
        patient_representation = patient_representation.mean(dim=1)
        
        proj_cl = self.cl_head(patient_representation)
        
        return logits_gene, patient_representation, proj_cl

# --- 5. 损失函数、训练与验证循环 ---
def nt_xent_loss(z1, z2, temperature=0.1):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    features = torch.cat([z1, z2], dim=0)
    labels = torch.cat([torch.arange(z1.shape[0]), torch.arange(z1.shape[0])], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z1.device)
    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
    labels, similarity_matrix = labels[~mask].view(labels.shape[0], -1), similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives, negatives = similarity_matrix[labels.bool()].view(labels.shape[0], -1), similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)
    return F.cross_entropy(logits / temperature, labels)

def get_model_inputs(batch, device):
    return (batch['gene_ids'].to(device),  batch['mut_type_ids'].to(device),
            batch['am_scores'].to(device), 
            batch['aa_wt_ids'].to(device),
            batch['aa_mt_ids'].to(device),
            batch['rel_positions'].to(device),
            batch['padding_mask'].to(device))

def train_epoch(model, dataloader, crit_mlm, optimizer, scheduler, device, config):
    model.train()
    total_loss, total_mlm_loss, total_cl_loss = 0, 0, 0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        inputs1 = get_model_inputs({k[:-2]: v for k, v in batch.items() if k.endswith('_1')}, device)
        inputs2 = get_model_inputs({k[:-2]: v for k, v in batch.items() if k.endswith('_2')}, device)
        logits_gene_1, _, proj_cl_1 = model(*inputs1)
        _, _, proj_cl_2 = model(*inputs2)
        loss_mlm = crit_mlm(logits_gene_1.view(-1, logits_gene_1.size(-1)), batch['labels_gene_1'].to(device).view(-1))
        loss_cl = nt_xent_loss(proj_cl_1, proj_cl_2, temperature=config['cl_temp'])
        loss = config['lambda_mlm'] * loss_mlm + config['lambda_cl'] * loss_cl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        
        total_loss += loss.item()
        total_mlm_loss += loss_mlm.item()
        total_cl_loss += loss_cl.item()
        pbar.set_postfix({'Loss': f"{loss.item():.3f}", 'MLM': f"{loss_mlm.item():.3f}", 'CL': f"{loss_cl.item():.3f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})
    
    avg_loss = total_loss / len(dataloader)
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    avg_cl_loss = total_cl_loss / len(dataloader)
    return avg_loss, avg_mlm_loss, avg_cl_loss

# --- NEW: Validation Epoch Function ---
def validate_epoch(model, dataloader, crit_mlm, device):
    model.eval()
    total_loss_sum = 0.0
    total_masked_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            inputs = get_model_inputs(batch, device)
            logits_gene, _, _ = model(*inputs)

            labels = batch['labels_gene'].to(device)
            loss = F.cross_entropy(
                logits_gene.view(-1, logits_gene.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            num_masked = (labels != -100).sum().item()
            total_loss_sum += loss.item()
            total_masked_tokens += num_masked

    return total_loss_sum / max(total_masked_tokens, 1)



# --- 6. 提取特征函数 (Unchanged) ---
def extract_final_features(model, samples, gene_le, mut_type_le, gene_ranks, device, config):
    print("\n--- Starting Final Feature Extraction ---")
    dataset = MutationSetDataset(samples, gene_le, mut_type_le, gene_ranks, perform_masking=False, perform_aug=False)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    all_embs, all_pids = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            _, patient_representation, _ = model(*get_model_inputs(batch, device))
            all_embs.append(patient_representation.cpu().numpy())
            all_pids.extend(batch['pid'])
    final_embs = np.vstack(all_embs)
    os.makedirs(config['output_dir'], exist_ok=True)
    np.save(os.path.join(config['output_dir'], 'patient_embeddings.npy'), final_embs)
    pd.DataFrame({'Tumor_Sample_Barcode': all_pids}).to_csv(os.path.join(config['output_dir'], 'patient_ids.tsv'), sep='\t', index=False)
    print(f"Features saved to {config['output_dir']}")

# --- 7. 主函数 (Main) (MODIFIED) ---
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    device = torch.device('npu' if torch.npu.is_available() else 'cpu')
    print(f"Using device: {device}")

    #预处理数据缓存
    cache_file = os.path.join("./", 'preprocessed_data.pkl')
    if os.path.exists(cache_file):
        print("Loading preprocessed data from cache...")
        train_samples, val_samples, gene_le, mut_type_le,gene_ranks = joblib.load(cache_file)
    else:
        # 执行原来的 prepare_data 逻辑
        train_samples, val_samples, gene_le, mut_type_le,gene_ranks = prepare_data(CONFIG)
        joblib.dump((train_samples, val_samples, gene_le, mut_type_le,gene_ranks), cache_file)
    
    
    joblib.dump(gene_le, os.path.join(CONFIG['output_dir'], 'gene_encoder.pkl'))
    
    joblib.dump(mut_type_le, os.path.join(CONFIG['output_dir'], 'mut_type_encoder.pkl'))
    
    
    gene_vocab_size, mut_type_vocab_size = \
        len(gene_le.classes_),  len(mut_type_le.classes_)
    print(f"Vocab Sizes: Gene={gene_vocab_size},  MutType={mut_type_vocab_size}")

    # --- DataLoaders Setup (MODIFIED) ---
    # Training DataLoader: with augmentation and masking
    train_ds = MutationSetDataset(train_samples, gene_le,  mut_type_le,gene_ranks, perform_masking=True, perform_aug=True)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True,drop_last=True)
    
    # Validation DataLoader (for loss calculation): with masking, NO augmentation
    val_ds_for_loss = MutationSetDataset(val_samples, gene_le,  mut_type_le, gene_ranks,perform_masking=True, perform_aug=False, fixed_masking=True,seed=CONFIG['seed'])
    val_loader_for_loss = DataLoader(val_ds_for_loss, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    

    model = MutationSetTransformer(CONFIG, gene_vocab_size,  mut_type_vocab_size)
    
    
    model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * len(train_loader) * CONFIG['epochs'], num_training_steps=len(train_loader) * CONFIG['epochs'])
    crit_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    
    # --- Training Loop (MODIFIED for validation loss-based early stopping) ---
    best_val_loss, patience_counter = float('inf'), 0
    print("\n--- Starting Training ---")
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        train_loss, train_mlm, train_cl = train_epoch(model, train_loader, crit_mlm, optimizer, scheduler, device, CONFIG)
        val_mlm_loss = validate_epoch(model, val_loader_for_loss, crit_mlm, device)
        
        
        print(f"  Train Loss: {train_loss:.4f} (MLM: {train_mlm:.4f}, CL: {train_cl:.4f})")
        print(f"  Val MLM Loss: {val_mlm_loss:.4f} ")
        
        if val_mlm_loss < best_val_loss:
            best_val_loss = val_mlm_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            print(f"  >>> Best model saved! New best validation MLM loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Early Stopping Counter: {patience_counter}/{CONFIG['patience']}")
        
        if patience_counter >= CONFIG['patience']:
            print("Early stopping triggered.")
            break
            
    print("\n--- Training Finished. Loading best model for feature extraction. ---")
    #model.load_state_dict(torch.load(os.path.join(CONFIG['output_dir'], 'best_model.pth')))
    # Note: Feature extraction should use all available labeled samples
    #extract_final_features(model, train_samples + val_samples, gene_le, mut_type_le, gene_ranks, device, CONFIG)

if __name__ == "__main__":
    main()