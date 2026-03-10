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
from PIL import Image, ImageFile
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

with open(config_path, 'r', encoding='utf-8') as f:
    yaml_config = yaml.safe_load(f)

CONFIG = yaml_config["CONFIG"]
AUG_STEPS = yaml_config["AUG_STEPS"]
PARAM_MAP = yaml_config["PARAM_MAP"]
FONT_NG_TRIGGERS = set(yaml_config["FONT_NG_TRIGGERS"])

# 동적 경로 및 워커 수 세팅
CONFIG["INPUT_ROOT"] = BASE_DIR
CONFIG["OUTPUT_ROOT"] = os.path.join(BASE_DIR, "Augmentation_output")

if CONFIG.get("NUM_WORKERS", 0) <= 0:
    TOTAL_CORES = multiprocessing.cpu_count()
    CONFIG["NUM_WORKERS"] = max(1, TOTAL_CORES - 2)


# =======================================================
# [2. 유틸리티 함수]
# =======================================================
def get_absolute_path(relative_path_from_csv):
    if not relative_path_from_csv: return None
    clean_rel = relative_path_from_csv.strip().replace('\\', '/').lstrip('./').lstrip('/')
    abs_path = os.path.join(CONFIG["INPUT_ROOT"], clean_rel)
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


def save_image_immediate(img, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    img.save(save_path, compress_level=1)
    return save_path


# =======================================================
# [3. 이미지 증강 엔진]
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
            axis_x = int(w * random.uniform(min_s, max_s))
            axis_y = int(h * random.uniform(min_s, max_s))
            color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            angle = random.randint(0, 360)
            cv2.ellipse(ink_layer, (cx, cy), (axis_x, axis_y), angle, 0, 360, color, -1)
            cv2.ellipse(alpha_mask, (cx, cy), (axis_x, axis_y), angle, 0, 360, 1.0, -1)

        k_size = (21, 21)
        ink_blurred = cv2.GaussianBlur(ink_layer, k_size, 0)
        mask_blurred = cv2.GaussianBlur(alpha_mask, k_size, 0)
        mask_blurred[mask_blurred < 0.05] = 0.0

        final_alpha = mask_blurred * global_opacity
        final_alpha_3ch = cv2.merge([final_alpha, final_alpha, final_alpha])
        image_float = image.astype(np.float32)
        ink_float = ink_blurred.astype(np.float32)
        output = image_float * (1.0 - final_alpha_3ch) + ink_float * final_alpha_3ch

        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), {"count": num_blobs,
                                                                           "opacity": round(global_opacity, 3)}

    @classmethod
    def apply_op(cls, img, tag, manual_param=None):
        if tag not in PARAM_MAP: return img, {}
        config = PARAM_MAP[tag]
        method = config["method"]
        processed = img.copy()
        params_log = {}

        if method == "shear":
            min_v, max_v = config["range"]
            if manual_param is not None:
                val, axis = manual_param
            else:
                val = random.uniform(min_v, max_v) * random.choice([-1, 1])
                axis = random.choice(['x', 'y'])
            processed = F.affine(processed, angle=0, translate=[0, 0], scale=1.0,
                                 shear=[val, 0.0] if axis == 'x' else [0.0, val],
                                 interpolation=v2.InterpolationMode.BILINEAR, fill=255)
            params_log = {"axis": axis, "val": round(val, 2)}

        elif method == "rotate":
            min_v, max_v = config["range"]
            val = manual_param if manual_param is not None else random.uniform(min_v, max_v) * random.choice([-1, 1])
            processed = F.affine(processed, angle=val, translate=[0, 0], scale=1.0, shear=[0.0, 0.0],
                                 interpolation=v2.InterpolationMode.BILINEAR, fill=255)
            params_log = {"angle": round(val, 2)}

        elif method == "perspective":
            val = random.uniform(config["range"][0], config["range"][1])
            w, h = processed.size
            startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
            endpoints = [[int(w * val), int(h * val)], [int(w * (1 - val)), int(h * val)],
                         [int(w * (1 - val)), int(h * (1 - val))], [int(w * val), int(h * (1 - val))]]
            processed = F.perspective(processed, startpoints, endpoints, interpolation=v2.InterpolationMode.BILINEAR,
                                      fill=config.get("fill", 255))
            params_log = {"dist": round(val, 3)}

        elif method == "elastic":
            alpha = random.uniform(config["alpha"][0], config["alpha"][1])
            sigma = random.uniform(config["sigma"][0], config["sigma"][1])
            processed = v2.ElasticTransform(alpha=alpha, sigma=sigma, fill=255)(processed)
            params_log = {"alpha": round(alpha, 1), "sigma": round(sigma, 1)}

        elif method == "hue":
            val = manual_param if manual_param is not None else random.uniform(config["range"][0],
                                                                               config["range"][1]) * random.choice(
                [-1, 1])
            val = max(-0.5, min(val, 0.5))
            processed = F.adjust_hue(processed, val)
            params_log = {"hue": round(val, 3)}

        elif method == "grayscale":
            val = manual_param if manual_param is not None else random.uniform(config["range"][0], config["range"][1])
            if random.random() < val: processed = F.to_grayscale(processed, num_output_channels=3)
            params_log = {"prob_used": round(val, 2)}

        elif method == "brightness":
            val = random.uniform(config["range"][0], config["range"][1])
            factor = 1.0 + (val * random.choice([-1, 1]))
            processed = F.adjust_brightness(processed, factor)
            params_log = {"factor": round(factor, 2)}

        elif method == "contrast":
            val = random.uniform(config["range"][0], config["range"][1])
            processed = F.adjust_contrast(processed, val)
            params_log = {"factor": round(val, 2)}

        elif method == "saturation":
            val = random.uniform(config["range"][0], config["range"][1])
            processed = F.adjust_saturation(processed, val)
            params_log = {"factor": round(val, 2)}

        elif method == "equalize":
            val = random.uniform(config["range"][0], config["range"][1])
            if random.random() < val: processed = F.equalize(processed)
            params_log = {"prob_used": round(val, 2)}

        elif method == "noise":
            val = random.uniform(config["range"][0], config["range"][1])
            processed, n_log = cls.apply_noise(processed, val)
            params_log.update(n_log)

        elif method == "stain":
            processed, s_log = cls.add_clean_stain(processed, config)
            params_log.update(s_log)

        return processed, params_log

    @staticmethod
    def generate_seed_param(tag):
        if tag not in PARAM_MAP: return None
        config = PARAM_MAP[tag]
        method = config["method"]
        if method == "shear":
            return (random.uniform(config["range"][0], config["range"][1]) * random.choice([-1, 1]),
                    random.choice(['x', 'y']))
        elif method == "rotate":
            return random.uniform(config["range"][0], config["range"][1]) * random.choice([-1, 1])
        elif method == "hue":
            return max(-0.5, min(random.uniform(config["range"][0], config["range"][1]) * random.choice([-1, 1]), 0.5))
        elif method == "grayscale":
            return random.uniform(config["range"][0], config["range"][1])
        return None


# =======================================================
# [4. 그룹 데이터 관리 및 워커 프로세스 (Race Condition 방지)]
# =======================================================
def get_target_subfolder(meta):
    s, c = meta['label_s'], meta['label_c']
    if s == 1 and c == 1: return "font_diff_letter_diff"
    if s == 1 and c == 0: return "font_diff_letter_same"
    if s == 0 and c == 1: return "font_same_letter_diff"
    return "font_same_letter_same"


def load_task_groups():
    csv_path = os.path.join(CONFIG["INPUT_ROOT"], CONFIG["TARGET_CSV"])
    if not os.path.exists(csv_path):
        csv_path = os.path.join(CONFIG["INPUT_ROOT"], "image_metadata", CONFIG["TARGET_CSV"])

    groups = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames: reader.fieldnames = [x.strip() for x in reader.fieldnames]

        for row in reader:
            real_ref = get_absolute_path(row.get('ref_path', ''))
            real_tar = get_absolute_path(row.get('tar_path', ''))

            if real_ref and real_tar:
                if real_ref not in groups:
                    groups[real_ref] = {
                        'ref_path': real_ref,
                        'ref_name': os.path.splitext(os.path.basename(real_ref))[0].replace(".png", ""),
                        'targets': []
                    }

                base_tgt_name = os.path.splitext(os.path.basename(real_tar))[0].replace(".png", "")
                if "@seg" in base_tgt_name:
                    base_tgt_name = base_tgt_name.split("@seg")[0] + "@seg"

                groups[real_ref]['targets'].append({
                    'tar_path': real_tar,
                    'tar_name': base_tgt_name,
                    'tar_filename_orig': os.path.basename(real_tar),
                    'meta': {
                        'font': row.get('font', ''), 'logo': row.get('logo', ''),
                        'label_s': int(row.get('label_s', 0)), 'label_c': int(row.get('label_c', 0))
                    }
                })
    return list(groups.values())


def worker_process(task_group):
    try:
        ref_path = task_group['ref_path']
        ref_name = task_group['ref_name']
        targets = task_group['targets']

        origin_ref_img = load_image_with_retry(ref_path)
        if not origin_ref_img: return [], []

        ref_cache = {}
        save_dir_ref = os.path.join(CONFIG["OUTPUT_ROOT"], "ref_img")

        ref_cache[()] = {
            "img": origin_ref_img,
            "saved_path": ref_path,
            "seed_param": None, "params_r": None
        }

        result_rows = []
        completed_tars = []

        for tgt_info in targets:
            origin_tgt_img = load_image_with_retry(tgt_info['tar_path'])
            if not origin_tgt_img: continue

            current_layer = [{
                "tgt_img": origin_tgt_img, "tgt_name": tgt_info['tar_name'],
                "meta": tgt_info['meta'], "ref_name": ref_name,
                "pair_history": ()
            }]

            for _, scope, methods in AUG_STEPS:
                next_layer = []

                for state in current_layer:
                    for tag in methods:
                        if state["meta"]['label_s'] == 1 and tag in FONT_NG_TRIGGERS: continue

                        next_meta = state["meta"].copy()
                        if next_meta['label_s'] == 0 and tag in FONT_NG_TRIGGERS:
                            next_meta['label_s'] = 1

                        next_pair_history = list(state["pair_history"])
                        if scope == 'pair': next_pair_history.append(tag)
                        next_pair_history = tuple(next_pair_history)

                        next_tgt_name = f"{state['tgt_name']}@{tag}"
                        next_ref_name = f"{state['ref_name']}@{tag}" if scope == 'pair' else state['ref_name']

                        if scope == 'pair':
                            if next_pair_history not in ref_cache:
                                seed_param = ImageAugmentor.generate_seed_param(tag)
                                parent_ref_img = ref_cache[state["pair_history"]]["img"]
                                res_ref_img, params_r = ImageAugmentor.apply_op(parent_ref_img, tag,
                                                                                manual_param=seed_param)

                                saved_ref_path = save_image_immediate(res_ref_img, save_dir_ref, f"{next_ref_name}.png")

                                ref_cache[next_pair_history] = {
                                    "img": res_ref_img, "saved_path": saved_ref_path,
                                    "seed_param": seed_param, "params_r": params_r
                                }

                            cached = ref_cache[next_pair_history]
                            seed_param = cached["seed_param"]
                            params_r = cached["params_r"]
                            final_ref_path = cached["saved_path"]

                            res_tgt_img, params_t = ImageAugmentor.apply_op(state["tgt_img"], tag,
                                                                            manual_param=seed_param)
                            aug_info = json.dumps({"ref": params_r, "tgt": params_t}, ensure_ascii=False)

                        else:
                            final_ref_path = ref_cache[state["pair_history"]]["saved_path"]
                            res_tgt_img, params_t = ImageAugmentor.apply_op(state["tgt_img"], tag)
                            aug_info = json.dumps({"ref": None, "tgt": params_t}, ensure_ascii=False)

                        subfolder = get_target_subfolder(next_meta)
                        save_dir_tgt = os.path.join(CONFIG["OUTPUT_ROOT"], "tar_img", subfolder)
                        saved_tgt_path = save_image_immediate(res_tgt_img, save_dir_tgt, f"{next_tgt_name}.png")

                        rel_ref = os.path.relpath(final_ref_path, CONFIG["OUTPUT_ROOT"]).replace(os.sep, '/')
                        rel_tgt = os.path.relpath(saved_tgt_path, CONFIG["OUTPUT_ROOT"]).replace(os.sep, '/')

                        result_rows.append({
                            'tar_path': f"./{rel_tgt}", 'ref_path': f"./{rel_ref}",
                            'font': next_meta.get('font', ''), 'logo': next_meta.get('logo', ''),
                            'label_s': next_meta['label_s'], 'label_c': next_meta['label_c'],
                            'aug_method': tag, 'aug_param': aug_info, 'label_stain': 1 if tag == 'stain_M' else 0
                        })

                        next_layer.append({
                            "tgt_img": res_tgt_img, "tgt_name": next_tgt_name,
                            "meta": next_meta, "ref_name": next_ref_name,
                            "pair_history": next_pair_history
                        })

                current_layer.clear()
                current_layer.extend(next_layer)

            completed_tars.append(tgt_info['tar_filename_orig'])

        ref_cache.clear()
        gc.collect()

        return result_rows, completed_tars

    except Exception:
        print(f"\n[Error] Group {task_group.get('ref_name')} failed.")
        traceback.print_exc()
        return [], []


# =======================================================
# [5. 메인 실행 함수]
# =======================================================
def main():
    multiprocessing.freeze_support()
    os.makedirs(CONFIG["OUTPUT_ROOT"], exist_ok=True)

    checkpoint_path = os.path.join(CONFIG["OUTPUT_ROOT"], CONFIG["CHECKPOINT_FILE"])
    output_csv = os.path.join(CONFIG["OUTPUT_ROOT"], CONFIG["OUTPUT_CSV"])

    completed_tasks = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            completed_tasks = set(line.strip() for line in f)

    all_groups = load_task_groups()

    task_list = []
    for group in all_groups:
        pending_targets = [t for t in group['targets'] if t['tar_filename_orig'] not in completed_tasks]
        if pending_targets:
            group['targets'] = pending_targets
            task_list.append(group)

    if CONFIG["IS_TEST_MODE"]:
        task_list = task_list[:CONFIG["TEST_COUNT"]]

    if not task_list:
        print("✅ 모든 작업이 완료되어 있습니다!")
        return

    headers = ['tar_path', 'ref_path', 'font', 'logo', 'label_s', 'label_c', 'aug_method', 'aug_param', 'label_stain']
    file_mode = 'a' if completed_tasks and os.path.exists(output_csv) else 'w'

    print(f"🚀 실행 워커 수: {CONFIG['NUM_WORKERS']} 개")

    with open(output_csv, file_mode, newline='', encoding='utf-8-sig') as out_f, \
            open(checkpoint_path, 'a', encoding='utf-8') as cp_f:

        writer = csv.DictWriter(out_f, fieldnames=headers)
        if file_mode == 'w': writer.writeheader()

        with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
            future_to_group = {executor.submit(worker_process, group): group['ref_name'] for group in task_list}

            for future in tqdm(as_completed(future_to_group), total=len(task_list), desc="데이터 증강 중"):
                try:
                    results, completed_tars = future.result()
                    if results:
                        writer.writerows(results)
                        out_f.flush()

                        for t_id in completed_tars:
                            cp_f.write(t_id + '\n')
                        cp_f.flush()
                except Exception:
                    # 필요시 로깅 추가 가능, 현재는 흐름 방해 방지를 위해 pass
                    pass


if __name__ == "__main__":
    main()