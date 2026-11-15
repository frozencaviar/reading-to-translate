import os
import json
import glob
from collections import defaultdict
import argparse

def parse_line(line, stseg_idx, token_idx):
    parts = line.strip().split('\t')
    if len(parts) <= max(stseg_idx, token_idx):
        return None, None
    seg_id = parts[stseg_idx]
    token = parts[token_idx].strip('"')
    if token == "---":
        return None, None
    try:
        seg_id = int(seg_id)
    except ValueError:
        pass
    return seg_id, token

def parse_table(table_path, is_tt=False):
    tokens_by_seg = defaultdict(list)
    with open(table_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return {}
            
        header = lines[0].strip().split('\t')
        has_header = False
        
        target_seg_col = "TTseg" if is_tt else "STseg"
        target_tok_col = "TToken" if is_tt else "SToken"
        
        stseg_idx = -1
        token_idx = -1
        
        if "SToken" in header or "TToken" in header or "Id" in header or "Study" in header:
            has_header = True
            # Find indices
            try:
                stseg_idx = header.index(target_seg_col) if target_seg_col in header else header.index("SGid" if not is_tt else "TGid")
            except ValueError:
                # fallback if naming is slightly different
                stseg_idx = [i for i, x in enumerate(header) if "seg" in x.lower() or "gid" in x.lower()]
                if not stseg_idx:
                    stseg_idx = 8 
                else:
                    stseg_idx = stseg_idx[0]
            
            try:
                token_idx = header.index(target_tok_col)
            except ValueError:
                if "Token" in header:
                    token_idx = header.index("Token")
                elif "SToken" in header and is_tt: # fallback
                    token_idx = header.index("SToken")
                elif "TToken" in header and not is_tt: 
                    token_idx = header.index("TToken")
                else:
                    token_idx = 9
        else:
            # Guessing based on common no-header layouts (like RUC17)
            # 0=Id, 1=Study, 2=Session, 3=SL, 4=TL, 5=Task, 6=TextType, 7=Part, 8=SegID, 9=Token
            stseg_idx = 8
            token_idx = 9
            
        start_idx = 1 if has_header else 0
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            seg_id, token = parse_line(line, stseg_idx, token_idx)
            if seg_id is not None and token is not None:
                tokens_by_seg[seg_id].append(token)
                
    return tokens_by_seg

def generate_translog(critt_dir, output_file):
    collections = defaultdict(list)
    
    st_files_direct = glob.glob(os.path.join(critt_dir, "*.st"))
    if st_files_direct:
        dirs_to_process = [("direct", critt_dir)]
    else:
        dirs_to_process = []
        studies = [d for d in os.listdir(critt_dir) if os.path.isdir(os.path.join(critt_dir, d))]
        for study in studies:
            table_dir = os.path.join(critt_dir, study, f"{study}-tables")
            if not os.path.exists(table_dir):
                table_dir = os.path.join(critt_dir, study)
            dirs_to_process.append((study, table_dir))
            
    for study, table_dir in dirs_to_process:
        st_files = glob.glob(os.path.join(table_dir, "*.st"))
        if not st_files:
            continue
            
        print(f"Processing directory: {table_dir} with {len(st_files)} sessions")

        for st_file in st_files:
            session_id = os.path.basename(st_file).replace('.st', '')
            tt_file = st_file.replace('.st', '.tt')
            if not os.path.exists(tt_file):
                print(f"Warning: missing tt file for {st_file}")
                continue
            
            # Read meta
            with open(st_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line: continue
                parts = first_line.strip().split('\t')
                
                # if header, next line might contain the values
                if "Study" in parts:
                    data_line = f.readline()
                    if data_line:
                        parts = data_line.strip().split('\t')
                
                if len(parts) < 6:
                    continue
                
                # Check column shift because of StudySession
                shift = 1 if "@" in parts[3] else 0
                
                study_val = parts[1]
                sess_val = parts[2]
                src_lang = parts[3 + shift]
                tgt_lang = parts[4 + shift]
                task_val = parts[5 + shift]
                
                task_map = {'T': 'translating', 'P': 'post-editing', 'C': 'copying'}
                task_str = task_map.get(task_val, task_val)
            
            st_dict = parse_table(st_file, is_tt=False)
            tt_dict = parse_table(tt_file, is_tt=True)
            
            src_tokens_list = []
            tgt_tokens_list = []
            
            all_segs = sorted(st_dict.keys(), key=lambda k: int(str(k).split('+')[0].split('-')[0].split('_')[0]) if any(c.isdigit() for c in str(k)) else 999999)
            
            for seg_id in all_segs:
                src_tokens_list.append(st_dict[seg_id])
                tgt_tokens_list.append(tt_dict.get(seg_id, []))
            
            collections[study_val].append({
                "study": study_val,
                "session_id": sess_val,
                "source": src_lang,
                "target": tgt_lang,
                "task": task_str,
                "tracker": "Unknown",
                "source_tokens": src_tokens_list,
                "target_tokens": tgt_tokens_list
            })
            
    with open(output_file, 'w', encoding='utf-8') as fw:
        json.dump(collections, fw, ensure_ascii=False)
        
    print(f"Successfully generated {output_file}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--critt_dir", type=str, required=True, help="Path to CRITT TPR-DB directory")
    parser.add_argument("--output", type=str, default="data/new_translog.json", help="Path to output json file")
    
    args = parser.parse_args()
    generate_translog(args.critt_dir, args.output)
