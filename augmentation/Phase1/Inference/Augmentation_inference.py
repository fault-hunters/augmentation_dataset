import os
import csv
import json
import random
import traceback
import multiprocessing
import gc
import time
import yaml
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps, ImageFile
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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
FONT_NG_TRIGGERS = set(yaml_config["FONT_NG_TRIGGERS"])

if CONFIG.get("NUM_WORKERS", 0) <= 0:
    TOTAL_CORES = multiprocessing.cpu_count()
    CONFIG["NUM_WORKERS"] = max(1, TOTAL_CORES - 2)


# =======================================================
# [2. 유틸리티 함수]
# =======================================================
def get_absolute_path(relative_path_from_csv, input_root):
    if not relative_path_from_csv: return None
    clean_rel = relative_path_from_csv.strip().replace('\\', '/').lstrip('./').lstrip('/')
    abs_path = os.path.join(input_root, clean_rel)
    return abs_path if os.path.exists(abs_path) else None


def load_image_with_retry(path, retries=3, delay=0.2):
    for i in range(retries):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            with Image.open(path) as f:
                img = f.convert("RGB")
                img.load()
                return img
        except Exception:
            if i == retries - 1: return None
            time.sleep(delay)
    return None


def save_inference_pair(ref_img, tgt_img, name_r, name_t, meta, aug_method, aug_params, output_root, rel_dir):
    dir_ref = os.path.join(output_root, CONFIG["SUB_REF"])
    dir_tgt = os.path.join(output_root, CONFIG["SUB_TAR"], rel_dir)

    os.makedirs(dir_ref, exist_ok=True)
    os.makedirs(dir_tgt, exist_ok=True)

    path_ref = os.path.join(dir_ref, f"{name_r}.png")
    path_tgt = os.path.join(dir_tgt, f"{name_t}.png")

    ref_img.save(path_ref, compress_level=1)
    tgt_img.save(path_tgt, compress_level=1)

    rel_ref = './' + os.path.relpath(path_ref, output_root).replace(os.sep, '/')
    rel_tgt = './' + os.path.relpath(path_tgt, output_root).replace(os.sep, '/')

    return {
        'tar_path': rel_tgt, 'ref_path': rel_ref,
        'font': meta.get('font', ''), 'logo': meta.get('logo', ''),
        'label_s': meta['label_s'], 'label_c': meta['label_c'],
        'label': meta.get('label', 0),
        'aug_method': aug_method, 'aug_param': aug_params,
        'label_stain': 1 if aug_method == 'stain_M' else 0
    }


# =======================================================
# [3. 이미지 변형 엔진]
# =======================================================
class ImageAugmentor:
    @staticmethod
    def apply_noise(img, severity_factor):
        img_tensor = v2.ToImage()(img)
        img_tensor = v2.ToDtype(torch.float32, scale=True)(img_tensor)
        noise = torch.randn_like(img_tensor) * severity_factor
        noisy_img = torch.clamp(img_tensor + noise, 0., 1.)
        return v2.ToPILImage()(noisy_img), {"severity": round(severity_factor, 4)}

    @staticmethod
    def add_clean_stain(pil_img, config):
        if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
        image = np.array(pil_img)
        h, w, _ = image.shape
        ink_layer = np.zeros((h, w, 3), dtype=np.uint8)
        alpha_mask = np.zeros((h, w), dtype=np.float32)
        num_blobs = random.randint(*config["count"])
        global_opacity = random.uniform(*config["opacity"])

        for _ in range(num_blobs):
            cx, cy = random.randint(0, w), random.randint(0, h)
            min_s, max_s = config["scale"]
            axis_x, axis_y = int(w * random.uniform(min_s, max_s)), int(h * random.uniform(min_s, max_s))
            color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            cv2.ellipse(ink_layer, (cx, cy), (axis_x, axis_y), random.randint(0, 360), 0, 360, color, -1)
            cv2.ellipse(alpha_mask, (cx, cy), (axis_x, axis_y), random.randint(0, 360), 0, 360, 1.0, -1)

        mask_blurred = cv2.GaussianBlur(alpha_mask, (21, 21), 0)
        mask_blurred[mask_blurred < 0.05] = 0.0
        f_alpha = cv2.merge([mask_blurred * global_opacity] * 3)
        out = image.astype(np.float32) * (1.0 - f_alpha) + cv2.GaussianBlur(ink_layer, (21, 21), 0).astype(
            np.float32) * f_alpha
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), {"count": num_blobs,
                                                                        "opacity": round(global_opacity, 3)}

    @classmethod
    def apply_op(cls, img, tag, manual_param=None):
        if tag not in PARAM_MAP: return img, {}
        cfg = PARAM_MAP[tag]
        method = cfg["method"]
        p = img.copy()
        log = {}

        if method == "shear":
            val, axis = manual_param if manual_param else (
                random.uniform(cfg["range"][0], cfg["range"][1]) * random.choice([-1, 1]), random.choice(['x', 'y']))
            p = F.affine(p, 0, [0, 0], 1.0, [val, 0] if axis == 'x' else [0, val], v2.InterpolationMode.BILINEAR, 255)
            log = {"axis": axis, "val": round(val, 2)}
        elif method == "rotate":
            val = manual_param if manual_param else random.uniform(cfg["range"][0], cfg["range"][1]) * random.choice([-1, 1])
            p = F.rotate(p, val, v2.InterpolationMode.BILINEAR, fill=255)
            log = {"angle": round(val, 2)}
        elif method == "perspective":
            sc = random.uniform(cfg["range"][0], cfg["range"][1])
            p = v2.RandomPerspective(distortion_scale=sc, p=1.0, fill=255)(p)
            log = {"distortion_scale": round(sc, 3)}
        elif method == "elastic":
            a, s = random.uniform(cfg["alpha"][0], cfg["alpha"][1]), random.uniform(cfg["sigma"][0], cfg["sigma"][1])
            p = v2.ElasticTransform(alpha=a, sigma=s)(p)
            log = {"alpha": round(a, 1), "sigma": round(s, 1)}
        elif method == "hue":
            val = manual_param if manual_param else max(-0.5, min(random.uniform(cfg["range"][0], cfg["range"][1]) * random.choice([-1, 1]), 0.5))
            p = F.adjust_hue(p, val)
            log = {"hue_factor": round(val, 3)}
        elif method == "grayscale":
            al = manual_param if manual_param else random.uniform(cfg["range"][0], cfg["range"][1])
            p = Image.blend(p.convert('RGB'), ImageOps.grayscale(p).convert("RGB"), al)
            log = {"gray_alpha": round(al, 2)}
        elif method == "brightness":
            f = max(0.0, 1.0 + (random.uniform(cfg["range"][0], cfg["range"][1]) * random.choice([-1, 1])))
            p = F.adjust_brightness(p, f)
            log = {"bright_factor": round(f, 2)}
        elif method == "contrast":
            v = random.uniform(cfg["range"][0], cfg["range"][1])
            p = F.adjust_contrast(p, v)
            log = {"contrast_factor": round(v, 2)}
        elif method == "saturation":
            v = random.uniform(cfg["range"][0], cfg["range"][1])
            p = F.adjust_saturation(p, v)
            log = {"sat_factor": round(v, 2)}
        elif method == "equalize":
            al = random.uniform(cfg["range"][0], cfg["range"][1])
            p = Image.blend(p.convert('RGB'), ImageOps.equalize(p.convert('RGB')), al)
            log = {"eq_alpha": round(al, 2)}
        elif method == "noise":
            p, log = cls.apply_noise(p, random.uniform(cfg["range"][0], cfg["range"][1]))
        elif method == "stain":
            p, log = cls.add_clean_stain(p, cfg)

        return p, log

    @staticmethod
    def generate_seed_param(tag):
        if tag not in PARAM_MAP: return None
        cfg = PARAM_MAP[tag]
        m = cfg["method"]
        if m == "shear": return (random.uniform(cfg["range"][0], cfg["range"][1]) * random.choice([-1, 1]), random.choice(['x', 'y']))
        if m == "rotate": return random.uniform(cfg["range"][0], cfg["range"][1]) * random.choice([-1, 1])
        if m == "hue": return max(-0.5, min(random.uniform(cfg["range"][0], cfg["range"][1]) * random.choice([-1, 1]), 0.5))
        if m == "grayscale": return random.uniform(cfg["range"][0], cfg["range"][1])
        return None


# =======================================================
# [4. 그룹 데이터 관리 및 워커 프로세스]
# =======================================================
def load_task_groups():
    groups = {}

    for csv_info in CONFIG["CSV_INFO_LIST"]:
        csv_path = csv_info["csv_path"]
        input_root = csv_info["input_root"]

        if not os.path.exists(csv_path):
            print(f"⚠️ [경고] CSV 파일을 찾을 수 없어 건너뜁니다: {csv_path}")
            continue

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                real_ref = get_absolute_path(row.get('ref_path', ''), input_root)
                real_tar = get_absolute_path(row.get('tar_path', ''), input_root)

                if real_ref and real_tar:
                    if real_ref not in groups:
                        groups[real_ref] = {
                            'ref_path': real_ref,
                            'ref_name': os.path.splitext(os.path.basename(real_ref))[0],
                            'targets': []
                        }

                    stem = os.path.splitext(os.path.basename(real_tar))[0]
                    base_tgt = stem.split("@seg")[0] + "@seg" if "@seg" in stem else stem

                    orig_tar_rel = row.get('tar_path', '').strip().replace('\\', '/').lstrip('./').lstrip('/')
                    rel_dir = os.path.dirname(orig_tar_rel)

                    groups[real_ref]['targets'].append({
                        'tar_path': real_tar,
                        'rel_dir': rel_dir,
                        'tar_name': base_tgt,
                        'tar_filename_orig': os.path.basename(real_tar),
                        'meta': {
                            'font': row.get('font', ''), 'logo': row.get('logo', ''),
                            'label_s': int(row.get('label_s', 0)), 'label_c': int(row.get('label_c', 0)),
                            'label': int(row.get('label', 0))
                        }
                    })

    return list(groups.values())


def worker_process(task_group, output_root):
    try:
        ref_path = task_group['ref_path']
        ref_name = task_group['ref_name']
        targets = task_group['targets']

        origin_ref_img = load_image_with_retry(ref_path)
        if not origin_ref_img: return [], []

        ref_cache = {(): {"img": origin_ref_img, "seed_param": None, "params_r": None}}

        result_rows = []
        completed_tars = []

        for tgt_info in targets:
            origin_tgt_img = load_image_with_retry(tgt_info['tar_path'])
            if not origin_tgt_img: continue

            row_origin = save_inference_pair(
                origin_ref_img, origin_tgt_img,
                f"{ref_name}@origin", f"{tgt_info['tar_name']}@origin",
                tgt_info['meta'], 'origin', '', output_root, tgt_info['rel_dir']
            )
            result_rows.append(row_origin)

            current_layer = [{
                "tgt_img": origin_tgt_img, "tgt_name": tgt_info['tar_name'],
                "meta": tgt_info['meta'], "ref_name": ref_name, "pair_history": ()
            }]

            for step_idx, scope, methods in AUG_STEPS:
                next_layer = []
                for state in current_layer:
                    for tag in methods:
                        if state["meta"]['label_s'] == 1 and tag in FONT_NG_TRIGGERS: continue

                        next_meta = state["meta"].copy()
                        if next_meta['label_s'] == 0 and tag in FONT_NG_TRIGGERS:
                            next_meta['label_s'] = 1
                            next_meta['label'] = 1

                        next_pair_history = list(state["pair_history"])
                        if scope == 'pair': next_pair_history.append(tag)
                        next_pair_history = tuple(next_pair_history)

                        next_tgt_name = f"{state['tgt_name']}@{tag}"
                        next_ref_name = f"{state['ref_name']}@{tag}" if scope == 'pair' else state['ref_name']

                        if scope == 'pair':
                            if next_pair_history not in ref_cache:
                                seed_param = ImageAugmentor.generate_seed_param(tag)
                                parent_ref_img = ref_cache[state["pair_history"]]["img"]
                                res_ref_img, params_r = ImageAugmentor.apply_op(parent_ref_img, tag, manual_param=seed_param)

                                ref_cache[next_pair_history] = {
                                    "img": res_ref_img, "seed_param": seed_param, "params_r": params_r
                                }

                            cached = ref_cache[next_pair_history]
                            current_ref_img = cached["img"]
                            res_tgt_img, params_t = ImageAugmentor.apply_op(state["tgt_img"], tag, manual_param=cached["seed_param"])
                            aug_info = json.dumps({"ref": cached["params_r"], "tgt": params_t})

                        else:
                            current_ref_img = ref_cache[state["pair_history"]]["img"]
                            res_tgt_img, params_t = ImageAugmentor.apply_op(state["tgt_img"], tag)
                            aug_info = json.dumps({"ref": None, "tgt": params_t})

                        row = save_inference_pair(
                            current_ref_img, res_tgt_img,
                            next_ref_name, next_tgt_name,
                            next_meta, tag, aug_info, output_root, tgt_info['rel_dir']
                        )
                        result_rows.append(row)

                        next_layer.append({
                            "tgt_img": res_tgt_img, "tgt_name": next_tgt_name,
                            "meta": next_meta, "ref_name": next_ref_name,
                            "pair_history": next_pair_history
                        })

                current_layer.extend(next_layer)

            completed_tars.append(tgt_info['tar_filename_orig'])

        ref_cache.clear()
        gc.collect()

        return result_rows, completed_tars

    except Exception:
        print(f"\n❌ [Error] Group {task_group.get('ref_name')} failed.")
        traceback.print_exc()
        return [], []


# =======================================================
# [5. 메인 실행 함수]
# =======================================================
def main():
    multiprocessing.freeze_support()

    print("=" * 60)
    print("🚀 데이터 증강 파이프라인(폴더 구조 유지형)을 시작합니다.")
    print("=" * 60)
    print("  [1] inference : 평가/추론용 데이터 증강")
    print("  [2] finetune  : 고혼진 맞춤형 파인튜닝 데이터 증강 (폴더 구조 보존)")
    print("=" * 60)

    while True:
        user_choice = input("👉 실행할 모드를 선택하세요 (1 또는 2 입력): ").strip().lower()
        if user_choice in ['1', 'inference']:
            mode = 'inference'
            break
        elif user_choice in ['2', 'finetune']:
            mode = 'finetune'
            break
        else:
            print("❌ 잘못된 입력입니다. '1' 또는 '2'를 입력해주세요.\n")

    selected_conf = MODE_CONFIG[mode]
    print(f"\n💡 [모드 설정] {mode.upper()} 모드로 세팅 중...")

    raw_csv_paths = selected_conf["TARGET_CSV"]
    if isinstance(raw_csv_paths, str):
        raw_csv_paths = [raw_csv_paths]

    csv_info_list = []
    for raw_csv in raw_csv_paths:
        expanded_csv = os.path.expanduser(raw_csv)
        if not os.path.isabs(expanded_csv):
            expanded_csv = os.path.join(BASE_DIR, expanded_csv)

        csv_info_list.append({
            "csv_path": expanded_csv,
            "input_root": os.path.dirname(expanded_csv)
        })

    CONFIG["CSV_INFO_LIST"] = csv_info_list
    CONFIG["OUTPUT_ROOT"] = os.path.join(BASE_DIR, selected_conf["OUTPUT_DIR"])
    CONFIG["OUTPUT_CSV"] = selected_conf["OUTPUT_CSV"]
    CONFIG["CHECKPOINT_FILE"] = selected_conf["CHECKPOINT_FILE"]

    os.makedirs(CONFIG["OUTPUT_ROOT"], exist_ok=True)
    out_csv = os.path.join(CONFIG["OUTPUT_ROOT"], CONFIG["OUTPUT_CSV"])
    cp_path = os.path.join(CONFIG["OUTPUT_ROOT"], CONFIG["CHECKPOINT_FILE"])

    all_groups = load_task_groups()
    if not all_groups:
        print(f"\n❌ 작업을 중단합니다. 정상적으로 로드된 CSV 데이터가 없습니다.")
        return

    done = set()
    if os.path.exists(cp_path):
        with open(cp_path, 'r', encoding='utf-8') as f: done = set(l.strip() for l in f)

    todo_groups = []
    for group in all_groups:
        pending = [t for t in group['targets'] if t['tar_filename_orig'] not in done]
        if pending:
            group['targets'] = pending
            todo_groups.append(group)

    if not todo_groups:
        print("\n✅ 모든 작업이 이미 완료되어 있습니다!")
        return

    headers = ['tar_path', 'ref_path', 'font', 'logo', 'label_s', 'label_c', 'label', 'aug_method', 'aug_param', 'label_stain']
    file_mode = 'a' if done and os.path.exists(out_csv) else 'w'

    print(f"\n🚀 실행 워커 수: {CONFIG['NUM_WORKERS']} 개")
    print(f"📦 총 {len(todo_groups)}개의 Reference 그룹 처리 중...")

    with open(out_csv, file_mode, newline='', encoding='utf-8-sig') as f, open(cp_path, 'a', encoding='utf-8') as cp:
        writer = csv.DictWriter(f, fieldnames=headers)
        if file_mode == 'w': writer.writeheader()

        with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as exe:
            futs = {exe.submit(worker_process, g, CONFIG["OUTPUT_ROOT"]): g['ref_name'] for g in todo_groups}
            for fut in tqdm(as_completed(futs), total=len(todo_groups), desc="증강 진행 중"):
                try:
                    res, completed_tars = fut.result()
                    if res:
                        writer.writerows(res)
                        f.flush()
                        for t_id in completed_tars: cp.write(t_id + '\n')
                        cp.flush()
                except Exception:
                    pass

    print(f"\n✨ [{mode.upper()}] 모드 증강 완료! 결과 폴더: {CONFIG['OUTPUT_ROOT']}")
    print(f"📄 통합 로그 파일: {out_csv}")


if __name__ == "__main__":
    main()