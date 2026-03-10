import os
import random
import multiprocessing
import yaml
import pandas as pd
import numpy as np
import torch
import json
from PIL import Image, ImageOps, ImageFile
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# [안전장치] 잘린 이미지 로드 방지
ImageFile.LOAD_TRUNCATED_IMAGES = False

# =======================================================
# [1. 설정(Config) 로드]
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config.yaml")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")

with open(config_path, 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)

CONFIG = yaml_config["CONFIG"]
MODE_CONFIG = yaml_config["MODE_CONFIG"]
AUG_STEPS = yaml_config["AUG_STEPS"]
PARAM_MAP = yaml_config["PARAM_MAP"]

if CONFIG.get("NUM_WORKERS", 0) <= 0:
    TOTAL_CORES = multiprocessing.cpu_count()
    CONFIG["NUM_WORKERS"] = max(1, TOTAL_CORES - 2)

# =======================================================
# [2. 유틸리티 함수]
# =======================================================
def load_image(path, is_mask=False):
    if not path or not os.path.exists(path): return None
    try:
        with Image.open(path) as f:
            img = f.convert("L" if is_mask else "RGB")
            img.load()
            return img
    except Exception:
        return None

# =======================================================
# [3. 이미지 변형 엔진 (Phase 2)]
# =======================================================
class ImageAugmentor:
    @classmethod
    def apply_op(cls, img, tag, param_map, seed=None, is_mask=False):
        if img is None: return None, {}
        p_cfg = param_map[tag]
        method = p_cfg["method"]
        processed = img.copy()
        interp = v2.InterpolationMode.NEAREST if is_mask else v2.InterpolationMode.BILINEAR
        
        if method == "shear":
            val, axis = seed
            processed = F.affine(processed, angle=0, translate=[0, 0], scale=1.0,
                                 shear=[val, 0.0] if axis == 'x' else [0.0, val],
                                 interpolation=interp, fill=0)
        elif method == "rotate":
            processed = F.rotate(processed, angle=seed, interpolation=interp, fill=0)
        elif method == "hue":
            if processed.mode == 'L': processed = processed.convert('RGB')
            processed = F.adjust_hue(processed, seed)
            if is_mask: processed = processed.convert('L')
        elif method == "grayscale":
            if processed.mode != 'RGB': processed = processed.convert('RGB')
            processed = Image.blend(processed, ImageOps.grayscale(processed).convert("RGB"), seed)
            if is_mask: processed = processed.convert('L')
        elif method == "brightness":
            processed = F.adjust_brightness(processed, seed)
        elif method == "contrast":
            processed = F.adjust_contrast(processed, seed)
        elif method == "equalize":
            if processed.mode != 'RGB': processed = processed.convert('RGB')
            processed = Image.blend(processed, ImageOps.equalize(processed), seed)
            if is_mask: processed = processed.convert('L')
        elif method == "noise":
            img_t = v2.ToDtype(torch.float32, scale=True)(v2.ToImage()(processed))
            noise = torch.randn_like(img_t) * seed
            processed = v2.ToPILImage()(torch.clamp(img_t + noise, 0., 1.))

        return processed, {"val": str(seed)}

    @staticmethod
    def generate_seed(tag, param_map):
        cfg = param_map[tag]
        m, r = cfg["method"], cfg["range"]
        if m == "shear": return (random.uniform(*r) * random.choice([-1, 1]), random.choice(['x', 'y']))
        if m in ["brightness", "contrast"]: return max(0.0, 1.0 + (random.uniform(*r) * random.choice([-1, 1])))
        return random.uniform(*r) * random.choice([-1, 1])

# =======================================================
# [4. 그룹 데이터 관리 및 워커 프로세스]
# =======================================================
def worker_process(task_data):
    row, output_root, split_folder = task_data
    
    current_layer = [{
        "r": load_image(row['ref']), "t": load_image(row['tar']),
        "rm": load_image(row['ref_m'], True), "tm": load_image(row['tar_m'], True),
        "method_chain": [],
        "param_chain": {}
    }]

    final_metadata = []

    for step_num, scope, methods in AUG_STEPS:
        next_layer = []
        for data in current_layer:
            if data['r'] is None or data['t'] is None: continue
            
            for tag in methods:
                seed = ImageAugmentor.generate_seed(tag, PARAM_MAP)
                apply_m = (step_num not in [3, 4]) 
                
                method_type = PARAM_MAP[tag]["method"]

                if scope == 'pair':
                    nr, _ = ImageAugmentor.apply_op(data['r'], tag, PARAM_MAP, seed)
                    nt, _ = ImageAugmentor.apply_op(data['t'], tag, PARAM_MAP, seed)
                    nrm, _ = ImageAugmentor.apply_op(data['rm'], tag, PARAM_MAP, seed, True) if (apply_m and data['rm']) else (data['rm'], None)
                    ntm, _ = ImageAugmentor.apply_op(data['tm'], tag, PARAM_MAP, seed, True) if (apply_m and data['tm']) else (data['tm'], None)
                else:
                    nr, nrm = data['r'], data['rm']
                    nt, _ = ImageAugmentor.apply_op(data['t'], tag, PARAM_MAP, seed)
                    ntm, _ = ImageAugmentor.apply_op(data['tm'], tag, PARAM_MAP, seed, True) if (apply_m and data['tm']) else (data['tm'], None)

                p_dict = {}
                if method_type == "shear":
                    val, axis = seed
                    p_dict = {"axis": axis, "val": round(val, 2)}
                elif method_type == "rotate":
                    p_dict = {"angle": round(seed, 2)}
                elif method_type == "hue":
                    p_dict = {"hue_factor": round(seed, 2)}
                elif method_type == "grayscale":
                    p_dict = {"gray_factor": round(seed, 2)}
                elif method_type == "brightness":
                    p_dict = {"bright_factor": round(seed, 2)}
                elif method_type == "contrast":
                    p_dict = {"contrast_factor": round(seed, 2)}
                elif method_type == "equalize":
                    p_dict = {"eq_factor": round(seed, 2)}
                elif method_type == "noise":
                    p_dict = {"noise_std": round(seed, 2)}
                else:
                    p_dict = {"val": str(seed)}

                step_key = f"{step_num}_{tag}"
                new_param_chain = data["param_chain"].copy()
                
                if scope == 'pair':
                    new_param_chain[step_key] = {"ref": p_dict, "tgt": p_dict}
                else:
                    new_param_chain[step_key] = {"ref": None, "tgt": p_dict}

                new_method_chain = data["method_chain"] + [tag]

                unique_id = f"{tag}_{random.getrandbits(16)}"
                base_name = f"{os.path.basename(row['tar']).split('.')[0]}_{unique_id}.png"
                
                path_r = os.path.join(output_root, split_folder, CONFIG["SUB_REF"], base_name)
                path_t = os.path.join(output_root, split_folder, CONFIG["SUB_TAR"], base_name)
                
                os.makedirs(os.path.dirname(path_r), exist_ok=True); nr.save(path_r)
                os.makedirs(os.path.dirname(path_t), exist_ok=True); nt.save(path_t)
                
                path_rm = None; path_tm = None
                if nrm:
                    path_rm = os.path.join(output_root, split_folder, CONFIG.get("SUB_REF_MASK", "ref_mask"), base_name)
                    os.makedirs(os.path.dirname(path_rm), exist_ok=True); nrm.save(path_rm)
                if ntm:
                    path_tm = os.path.join(output_root, split_folder, CONFIG.get("SUB_TAR_MASK", "tar_mask"), base_name)
                    os.makedirs(os.path.dirname(path_tm), exist_ok=True); ntm.save(path_tm)

                next_layer.append({
                    "r": nr, "t": nt, "rm": nrm, "tm": ntm, 
                    "method_chain": new_method_chain,
                    "param_chain": new_param_chain, 
                    "res_r": path_r, "res_t": path_t, "res_rm": path_rm, "res_tm": path_tm
                })
                
                final_metadata.append({
                    "type": split_folder, 
                    "tar_path": path_t,
                    "ref_path": path_r,
                    "tar_mask_path": path_tm,
                    "ref_mask_path": path_rm,
                    "aug_method": " > ".join(new_method_chain),
                    "aug_param": json.dumps(new_param_chain)
                })
        
        current_layer = next_layer
    
    return final_metadata

# =======================================================
# [5. 메인 실행 함수]
# =======================================================
def main():
    multiprocessing.freeze_support()
    
    print("=" * 60)
    print("🚀 데이터 증강 파이프라인(Phase 2)을 시작합니다. (Train/Val 독립 CSV 생성 모드)")
    print("=" * 60)
    
    mode = "phase2"
    selected_conf = MODE_CONFIG[mode]
    
    output_root = os.path.join(BASE_DIR, selected_conf["OUTPUT_DIR"])
    os.makedirs(output_root, exist_ok=True)
    
    all_tasks = []
    input_root = CONFIG["INPUT_ROOT"]
    
    for src in selected_conf["DATA_SOURCES"]:
        csv_path = os.path.join(BASE_DIR, src["path"])
        if not os.path.exists(csv_path):
            print(f"⚠️ [경고] CSV 파일을 찾을 수 없어 건너뜁니다: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        m = selected_conf["COLUMN_MAPS"][src["type"]]
        
        split_folder = src.get("split", "train")
        
        for _, r in df.iterrows():
            all_tasks.append(({
                'ref': os.path.join(input_root, str(r[m.get('ref')])) if pd.notna(r.get(m.get('ref'))) else None,
                'tar': os.path.join(input_root, str(r[m.get('tar')])) if pd.notna(r.get(m.get('tar'))) else None,
                'ref_m': os.path.join(input_root, str(r[m.get('ref_m')])) if m.get('ref_m') and pd.notna(r.get(m.get('ref_m'))) else None,
                'tar_m': os.path.join(input_root, str(r[m.get('tar_m')])) if m.get('tar_m') and pd.notna(r.get(m.get('tar_m'))) else None
            }, output_root, split_folder))

    if not all_tasks:
        print("❌ 작업을 진행할 데이터가 없습니다.")
        return

    print(f"🚀 총 {len(all_tasks)}개의 원본 쌍에 대해 증강 조합 생성을 시작합니다.")
    print(f"⚙️ 실행 워커 수: {CONFIG['NUM_WORKERS']} 개")
    
    all_results = []
    with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as ex:
        for res_list in tqdm(ex.map(worker_process, all_tasks), total=len(all_tasks)):
            if res_list:
                all_results.extend(res_list)

    if all_results:
        # 결과를 DataFrame으로 변환
        df_all = pd.DataFrame(all_results)
        
        print(f"✅ 완료! 총 {len(all_results)}개의 증강 데이터 로그가 저장되었습니다.")
        
        # 👉 type(split_folder)을 기준으로 폴더별로 각각 CSV를 생성하여 저장
        for split_type, group_df in df_all.groupby('type'):
            # 저장할 폴더 (예: phase2_augmented_results/train)
            split_dir = os.path.join(output_root, split_type)
            os.makedirs(split_dir, exist_ok=True)
            
            # 💡 파일 이름을 {split_type}_aug_log.csv 형태로 변경
            log_path = os.path.join(split_dir, f"{split_type}_aug_log.csv") 
            
            # CSV 저장 (index 제외)
            group_df.to_csv(log_path, index=False)
            
            print(f"📍 [{split_type.upper()}] 로그 위치: {log_path} ({len(group_df)} 건)")
    else:
        print("❌ 생성된 데이터가 없습니다. 경로 설정을 확인해주세요.")

if __name__ == "__main__":
    main()