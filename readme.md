# Image Synthesis & Augmentation

**폰트 렌더링 → 가상 화장품 제품 이미지 합성 → 데이터 증강** 순의 데이터 생성 파이프라인

---

## 개요

| 구분 | 모듈 (파일명) | 주요 기능 및 역할 |
| --- | --- | --- |
| **Font** | `Font.ipynb` | TTF 폰트 파일을 사용하여 전면/후면 라벨(흑백/컬러) 이미지를 생성 |
| **Dataset Generator** | `Dataset.ipynb` | Gemini 3 API(Nano banana pro)를 활용해 용기(Ref A), 로고(Ref B), 라벨(Ref C)을 조합한 가상의 화장품 제품 이미지를 생성 |
| **Data Augmentation** | `Augmentation_*.py` | 생성된 이미지에 노이즈, 회전, 얼룩 등 다양한 변형을 가하여 학습용 데이터셋을 생성 |

---

## 상세 설명

### 1️. Font (`Font.ipynb`)

* **라벨 생성:** JSONL(전면) 및 CSV(후면) 텍스트를 읽어 폰트별 라벨 이미지를 자동 생성

(데이터는 AI Hub에서 제공한 "의약품, 화장품 패키징 OCR 데이터"를 참조하였습니다.)
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EC%9D%98%EC%95%BD%ED%92%88,%20%ED%99%94%EC%9E%A5%ED%92%88%20%ED%8C%A8%ED%82%A4%EC%A7%95%20OCR%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=633

* **TTF 파일 검증 :** 폰트 내 글리프(Glyph) 누락 여부를 체크하고 손상된 텍스트 렌더링을 방지
* **출력:** 1024x1024 크기의 정사각 캔버스에 최적화된 텍스트 이미지를 생성

### 2️. Dataset Generator (`Dataset.ipynb`)

* **Gemini 3 Pro API:** 멀티모달 프롬프트를 통해 참조 이미지들의 구조와 디자인을 유지하며 합성
* **병렬 처리:** `ThreadPoolExecutor`를 사용하여 여러 개의 이미지를 동시에 생성하여 작업 처리 속도를 높였습니다.
* **로그 관리:** 생성 이미지의 정보를 `generation_log.csv`에 기록

### 3️. Data Augmentation 

Torchvision 패키지의 Augmix 라이브러리를 사용하였습니다.
(https://arxiv.org/abs/1912.02781, https://docs.pytorch.org/vision/main/generated/torchvision.transforms.AugMix.html)

* **학습 (`_train`):** 라벨 조합 (텍스트 NG, Glyph NG)에 따라 폴더를 자동 분류하여 저장
* **추론 (`_inference`):** 평가용 데이터셋을 생성
* **최종 (`Phase2_augmentation`):** 프로젝트의 PoC Test를 위한, 마스크(Mask) 이미지까지 포함된 증강

![광고 텍스트 불량 감지 시스템 (SKP ASAC) Final](https://github.com/user-attachments/assets/5fa96dbc-32d2-435e-80ce-910dd53ec780)

| 분류 | 적용 기법 | 상세 내용 |
| --- | --- | --- |
| **기하학적 변형** | **Shear, Rotate, Perspective, Elastic** | 이미지의 기울기, 회전, 원근감 및 탄성 변형을 적용하여 촬영 환경의 다양성을 확보 |
| **색상 및 대비** | **Hue, Brightness, Contrast, Saturation, Equalize** | 색조, 밝기, 대비, 채도를 조절하여 조명 환경 및 카메라 센서의 변화를 재현 |
| **화질 및 노이즈** | **Gaussian Noise, Grayscale** | 저화질이나 노이즈가 심한 촬영 환경을 재현 |
| **특수 효과** | **Stain (Ink Blobs)** | **OpenCV**를 활용해 이미지 위에 무작위 오염(텍스트 가려짐) 효과를 생성 |

---

## 데이터 레이블링 및 분류

증강된 데이터셋은 `label_s`와 `label_c`의 조합에 따라 4개의 하위 폴더로 분류되며, 특정 증강 기법 적용 여부에 따라 레이블이 변경됩니다.

### 1. 레이블 정의

| 레이블 | 값 | 의미 | 비고 |
| --- | --- | --- | --- |
| **label_s** | `0` | **Same Style**: 참조(Ref)와 대상(Tar)의 폰트가 동일함 | - |
|  | `1` | **Different Style**: 참조와 대상의 폰트가 다름 | 특정 증강 시 `1`로 기록 |
| **label_c** | `0` | **Same Content**: 두 이미지의 글자(문자) 내용이 동일함 | - |
|  | `1` | **Different Content**: 두 이미지의 글자 내용이 다름 | - |
| **label_stain** | `0` | **Clean**: Stain 증강이 적용되지 않은 깨끗한 상태 | - |
|  | `1` | **Stained**: Stain 증강이 적용되어 얼룩이 존재함 | `stain_M` 증강 |

---

### 2. 폴더 구조 (Subfolder Mapping)

Augmentation이 된 이미지는 `label_s`와 `label_c`의 조합($2 \times 2$)에 따라 `tar_img` 내의 저장 경로가 결정됩니다.

| label_s | label_c | 저장 폴더명 (`tar_img/` 하위 폴더명) | 설명 |
| --- | --- | --- | --- |
| $0$ | $0$ | `font_same_letter_same` | 동일 폰트 + 동일 글자 |
| $0$ | $1$ | `font_same_letter_diff` | 동일 폰트 + 다른 글자 |
| $1$ | $0$ | `font_diff_letter_same` | 다른 폰트 + 동일 글자 |
| $1$ | $1$ | `font_diff_letter_diff` | 다른 폰트 + 다른 글자 |

---

### 3. 레이블 변경 로직

증강 과정에서 원본 메타데이터의 레이블이 아래의 규칙에 따라 수정됩니다.

* **폰트 왜곡**: `FONT_NG_TRIGGERS`에 정의된 증강 기법이 적용될 경우, 기존에 `label_s: 0` (동일 폰트)이었던 데이터는 CSV 출력에 `label_s: 1` (다른 폰트/스타일)로 기록됩니다.

* **Stain**: `aug_method`가 `stain_M`일 경우에 `label_stain` 값이 `1`로 기록됩니다.
  
---

### 4. CSV 출력

최종 생성되는 `Augmentation_output.csv`의 데이터 구조입니다.

* `tar_path` / `ref_path`: 이미지들의 상대 경로
* `font` / `logo`: 원본 속성 정보 (사용한 폰트, 로고 이미지 정보)
* `label_s` / `label_c`: 폰트 스타일 및 내용 일치 여부
* `aug_method`: 적용된 증강 기법 명칭
* `aug_param`: 적용된 파라미터 값
* `label_stain`: stain 적용 여부

---

