# WaveUI-25K 데이터셋 분석 요약

## 1. 개요

- **데이터셋 이름:** `agentsea/wave-ui-25k`
- **데이터셋 크기:** 24,978개 샘플
- **데이터 분할:** `train` 단일 분할로 구성

---

## 2. 피처 (Features) 상세

데이터셋은 UI 스크린샷과 함께 각 요소에 대한 상세한 정보를 포함하고 있습니다. 각 샘플은 다음 13개의 피처로 구성됩니다.

| 피처 명 | 데이터 타입 | 설명 | 예시 |
|---|---|---|---|
| `image` | PIL Image | UI 스크린샷 이미지 | `<PIL.PngImagePlugin.PngImageFile image...>` |
| `instruction` | string | 상호작용할 UI 요소의 타입 | `StaticText, link` |
| `bbox` | list[float] | UI 요소의 경계 상자 `[x_min, y_min, x_max, y_max]` | `[1023.0, 55.0, 1092.0, 75.0]` |
| `resolution` | list[int] | 원본 이미지의 해상도 `[width, height]` | `[1280, 720]` |
| `source` | string | 데이터 출처 | `webui` |
| `platform` | string | UI 플랫폼 (e.g., web, mobile) | `web` |
| `name` | string | UI 요소의 이름 | `about us button` |
| `description` | string | UI 요소에 대한 상세 설명 | `Text link with the words '''About Us'''` |
| `type` | string | UI 요소의 종류 | `link` |
| `OCR` | string | OCR로 추출한 요소의 텍스트 | `About Us` |
| `language` | string | UI 텍스트의 언어 | `English` |
| `purpose` | string | 요소의 목적 또는 기능 | `navigate to the About Us section...` |
| `expectation` | string | 요소와 상호작용 시 예상되는 결과 | `the About Us section will load` |

---

## 3. 초기 관찰 및 활용 방안

- **State 표현:** `image`와 `bbox` 정보를 사용해 UI의 시각적 상태(visual state)를 정의할 수 있습니다. 추가적으로 `OCR`, `description`, `type` 등의 정보를 활용하여 의미론적 상태(semantic state)를 구성할 수 있습니다.
- **Action 정의:** `instruction` 또는 `purpose`를 기반으로 RL 에이전트가 수행할 수 있는 행동(action)을 정의하는 데 활용할 수 있습니다.
- **Reward 설계:** 현재 데이터셋에는 직접적인 사용자 반응(클릭률 등) 데이터가 없습니다. 프로젝트 계획에 따라, 이 UI 데이터와 별도의 사용자 반응 데이터셋(예: RetailRocket)을 연결하여 보상 예측 모델(Reward Simulator)을 구축해야 합니다.

이 분석은 프로젝트의 다음 단계인 'State 정의를 위한 전처리'의 기반이 될 것입니다.

---

## 4. 데이터셋 상세 정보 (Web-Researched)

### 4.1. agentsea/wave-ui-25k

- **개요:** AI 에이전트가 UI를 이해하고 상호작용하는 방법을 훈련시키기 위해 설계된 25,000개의 레이블이 지정된 UI 요소 예제 모음입니다. Hugging Face에서 사용할 수 있으며, 여러 소스 데이터셋을 결합하고 중복, 겹치는 데이터 및 저품질 예제를 필터링하여 고품질 컬렉션을 보장합니다.
- **주요 특징:**
    - **풍부한 주석:** 각 UI 요소에 대해 상세한 설명, 요소 유형, OCR 텍스트, 언어, 일반적인 목적 및 클릭 시 예상되는 결과에 대한 주석이 포함됩니다.
    - **데이터 형식:** 데이터는 Parquet 형식으로 저장됩니다.
- **활용:** 이 데이터셋은 그래픽 사용자 인터페이스에서 작업을 이해하고 자동화하는 멀티모달 모델의 기능을 개선하기 위한 연구에 사용됩니다.

### 4.2. retailrocket/ecommerce-dataset

- **개요:** 실제 전자상거래 웹사이트에서 수집된 데이터로, 특히 암시적 피드백을 사용하는 추천 시스템 연구를 장려하기 위해 공개되었습니다. 모든 데이터는 기밀성을 위해 해시 처리되었습니다.
- **주요 파일 구성:**
    - **`events.csv`:** 'view', 'addtocart', 'transaction'과 같은 사용자 상호작용 데이터를 포함합니다. 약 140만 명의 순 방문자로부터 4.5개월 동안 270만 개 이상의 이벤트가 포함됩니다.
    - **`item_properties.csv`:** 약 417,000개의 고유한 아이템에 대한 2,000만 개 이상의 속성을 포함합니다. 가격과 같은 속성은 변경될 수 있으므로 데이터에 타임스탬프가 포함됩니다.
    - **`category_tree.csv`:** 아이템의 카테고리 계층 구조를 설명합니다.
- **활용:** 이 데이터셋은 세션 기반 추천 시스템을 구축하는 등의 작업에 자주 사용됩니다.
