# Image Synthesis & Augmentation

**폰트 렌더링 → 가상 화장품 제품 이미지 합성 → 데이터 증강**으로 이어지는 데이터 생성 파이프라인

---

## 개요

| 구분 | 모듈 (파일명) | 주요 기능 및 역할 |
| --- | --- | --- |
| **Font** | `Font.ipynb` | TTF 폰트 파일을 사용하여 전면/후면 라벨(흑백/컬러) 이미지를 생성 |
| **Dataset Generator** | `Dataset.ipynb` | Gemini 3(Nano banana pro)를 활용해 용기(Ref A), 로고(Ref B), 라벨(Ref C)을 조합한 가상의 화장품 제품 이미지를 생성 |
| **Data Augmentation** | `Augmentation_*.py` | 생성된 이미지에 노이즈, 회전, 얼룩 등 다양한 변형을 가하여 학습용 데이터셋을 생성 |

---

## 상세 설명

### 1️. Font (`Font.ipynb`)

* **라벨 생성:** JSONL(전면) 및 CSV(후면) 텍스트를 읽어 폰트별 라벨 이미지를 자동 생성

(데이터는 AI Hub에서 제공한 "의약품, 화장품 패키징 OCR 데이터"를 참조하였습니다.)
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EC%9D%98%EC%95%BD%ED%92%88,%20%ED%99%94%EC%9E%A5%ED%92%88%20%ED%8C%A8%ED%82%A4%EC%A7%95%20OCR%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=633

* **검증 로직:** 폰트 내 글리프(Glyph) 누락 여부를 체크하고 손상된 텍스트 렌더링을 방지
* **출력:** 1024x1024 크기의 정사각 캔버스에 최적화된 텍스트 이미지를 생성

### 2️. Dataset Generator (`Dataset.ipynb`)

* **Gemini 3 Pro API:** 멀티모달 프롬프트를 통해 참조 이미지들의 구조와 디자인을 유지하며 합성
* **병렬 처리:** `ThreadPoolExecutor`를 사용하여 여러 개의 이미지를 동시에 생성하여 작업 처리 속도를 높였습니다.
* **로그 관리:** 생성 성공/실패 여부를 `generation_log.csv`에 기록

### 3️. Data Augmentation

* **학습용 (`_train`):** 라벨 조합 (텍스트 NG, Glyph NG)에 따라 폴더를 자동 분류하여 저장
* **추론용 (`_inference`):** 평가용 데이터셋을 생성
* **단계별 (`Phase2_augmentation`):** 프로젝트의 PoC Test를 위한, 마스크(Mask) 이미지까지 포함된 증강
---

