# README

## 개요

이 저장소는 Replay-PAD 기반 1-frame Face Anti-Spoofing 실험 코드를 정리한 것이다.
코드는 데이터 준비, 1-frame 추출, 모델 학습, threshold 선택, test 평가, 결과 비교의 순서로 구성된다.

---

## 전체 흐름

전체 흐름은 아래 4단계로 보면 된다.

1. `metadata/`에서 전체 비디오 목록 확인
2. `frame_index/`에서 각 비디오에 대응되는 1-frame 확인
3. `src/`에서 데이터 준비, 모델 정의, 학습, 평가 코드 확인
4. `outputs/results/`에서 최종 성능과 비교 결과 확인

---

## 폴더 구조와 역할

| 폴더               | 설명                                                     |
| ------------------ | -------------------------------------------------------- |
| `src/`             | 데이터 준비, 데이터셋 로딩, 모델 정의, 학습 및 평가 코드 |
| `metadata/`        | 원본 비디오 기준 메타데이터 CSV                          |
| `frame_index/`     | 각 비디오에서 선택된 1-frame 정보 CSV                    |
| `outputs/results/` | 학습 로그, 평가 결과, subgroup 분석, 모델 비교표         |

---

## 주요 파일 설명

### `metadata/`

| 파일                      | 설명                                                                                            |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| `replay_pad_metadata.csv` | 전체 비디오 목록과 `video_id`, `label`, `attack_type`, `split`, `video_path`를 정리한 기준 파일 |

### `frame_index/`

| 파일                    | 설명                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `replay_pad_1frame.csv` | 각 비디오에서 추출된 1장의 프레임 경로와 `frame_idx`를 기록한 파일 |

### `src/prepare`

| 파일                           | 설명                                                                        |
| ------------------------------ | --------------------------------------------------------------------------- |
| `build_replay_pad_metadata.py` | `data/train`, `data/devel`, `data/test`를 순회하여 메타데이터 CSV 생성      |
| `extract_one_frame.py`         | 메타데이터를 기준으로 각 비디오에서 1-frame을 추출하고 frame index CSV 생성 |

### `src/datasets`

| 파일                          | 설명                                                |
| ----------------------------- | --------------------------------------------------- |
| `replay_pad_frame_dataset.py` | `replay_pad_1frame.csv`를 읽어 split별 dataset 구성 |

### `src/models`

| 파일                            | 설명                                       |
| ------------------------------- | ------------------------------------------ |
| `resnet18_baseline.py`          | ResNet18 기반 2-class 분류기 정의          |
| `mobilenetv3_small_baseline.py` | MobileNetV3-Small 기반 2-class 분류기 정의 |

### `src/engine`

| 파일                                   | 설명                                                                       |
| -------------------------------------- | -------------------------------------------------------------------------- |
| `train_resnet18_1frame.py`             | ResNet18 학습, best checkpoint 및 train log 저장                           |
| `evaluate_resnet18_1frame.py`          | ResNet18 평가, devel threshold 선택, test frame/video metric 계산          |
| `train_mobilenetv3_small_1frame.py`    | MobileNetV3-Small 학습, best checkpoint 및 train log 저장                  |
| `evaluate_mobilenetv3_small_1frame.py` | MobileNetV3-Small 평가, devel threshold 선택, test frame/video metric 계산 |
| `pad_metrics.py`                       | APCER, BPCER, ACER, HTER 계산, threshold 탐색, video-level 집계            |
| `make_model_comparison_table.py`       | 각 모델의 test video-level 결과를 모아 비교표 생성                         |

### `outputs/results/`

| 파일                                         | 설명                                           |
| -------------------------------------------- | ---------------------------------------------- |
| `resnet18_1frame_train_log.json`             | ResNet18 학습 설정과 epoch별 로그              |
| `mobilenetv3_small_1frame_train_log.json`    | MobileNetV3-Small 학습 설정과 epoch별 로그     |
| `resnet18_eval_results.json`                 | ResNet18의 threshold와 test 성능 요약          |
| `mobilenetv3_small_eval_results.json`        | MobileNetV3-Small의 threshold와 test 성능 요약 |
| `resnet18_test_frame_subgroups.csv`          | ResNet18의 frame-level subgroup 결과           |
| `resnet18_test_video_subgroups.csv`          | ResNet18의 video-level subgroup 결과           |
| `mobilenetv3_small_test_frame_subgroups.csv` | MobileNetV3-Small의 frame-level subgroup 결과  |                   
| `mobilenetv3_small_test_video_subgroups.csv` | MobileNetV3-Small의 video-level subgroup 결과  |
| `model_comparison_video_level.csv`           | 모델 간 최종 video-level 비교표                |

---

## train / evaluate 흐름

### Train

- 입력은 `frame_index/replay_pad_1frame.csv` 이다.
- 이 CSV에서 `split == train` 인 row만 읽어 학습 데이터로 사용하고, `split == devel` 인 row는 validation 용도로 사용한다.
- 각 row에서 `frame_path` 로 이미지를 불러오고, `label` 값을 `real=0`, `attack=1` 로 변환해 2-class 분류 학습을 수행한다.
- 학습 중에는 devel loss를 기준으로 best checkpoint를 저장한다.

### Evaluate

- 평가도 동일하게 `frame_index/replay_pad_1frame.csv` 를 기준으로 한다.
- `split == devel` 인 샘플에 대해 spoof score를 예측한 뒤, 이 점수를 사용해 threshold를 선택한다.
- 이후 선택된 threshold를 고정한 채 `split == test` 샘플에 적용한다.
- test에서는 먼저 각 샘플의 frame-level score와 label로 metric을 계산한다.
- 이후 같은 `video_id`를 기준으로 score를 집계해 video-level metric을 다시 계산한다.
- 이번 실험은 비디오당 1-frame만 사용하므로, frame-level 결과와 video-level 결과는 동일하게 나온다.
- 최종 비교에는 `test_video_metrics` 를 사용한다.

---

## 읽는 순서

아래 순서로 보면 전체 구조를 빠르게 파악할 수 있다.

1. `metadata/replay_pad_metadata.csv`
2. `frame_index/replay_pad_1frame.csv`
3. `src/prepare/build_replay_pad_metadata.py`
4. `src/prepare/extract_one_frame.py`
5. `src/datasets/replay_pad_frame_dataset.py`
6. `src/models/resnet18_baseline.py`
7. `src/models/mobilenetv3_small_baseline.py`
8. `src/engine/train_resnet18_1frame.py`
9. `src/engine/train_mobilenetv3_small_1frame.py`
10. `src/engine/pad_metrics.py`
11. `src/engine/evaluate_resnet18_1frame.py`
12. `src/engine/evaluate_mobilenetv3_small_1frame.py`
13. `src/engine/make_model_comparison_table.py`
14. `outputs/results/model_comparison_video_level.csv`

---

## 실행 흐름 요약

- 메타데이터 생성
- 1-frame 추출
- 모델 학습
- devel에서 threshold 선택
- test에서 frame-level / video-level 평가
- 결과 저장 및 모델 비교표 생성

---
