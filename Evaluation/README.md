# Evaluation

`Evaluation` 폴더는 휴리스틱 정책(`R1`, `R3`, `R5`)과 사전학습 모델(`PPO`, `ProMP`)의 총비용(`Total Cost`)을 비교하기 위한 독립 실행 코드입니다.

## 실행

프로젝트 루트(`AIIS_META_RL`)에서:

```powershell
python -m Evaluation.run_policy_comparison
```

옵션 예시:

```powershell
python -m Evaluation.run_policy_comparison --days 200 --stationary-num-scenarios-per-seed 100 --nonstationary-num-sequences-per-seed 100
```

멀티시드 예시:

```powershell
python -m Evaluation.run_policy_comparison --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 --stationary-num-scenarios-per-seed 100 --nonstationary-num-sequences-per-seed 100
```

고정 시나리오 모드 예시(Few_shot_learning의 `FIXED_SCENARIO_CASES` 5개 사용):

```powershell
python -m Evaluation.run_policy_comparison --scenario-mode fixed --seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 --nonstationary-num-sequences-per-seed 5
```

## 설정 파일

`Evaluation/config.py`에서 아래 항목을 수정할 수 있습니다.

- `PRETRAINED_PROMP_MODEL_PATH`
- `PRETRAINED_PPO_MODEL_PATH`
- `EVALUATION_SEEDS`
- `SCENARIO_MODE`
- `STATIONARY_NUM_SCENARIOS_PER_SEED`
- `NONSTATIONARY_NUM_SEQUENCES_PER_SEED`
- `NONSTATIONARY_SEGMENTS`
- `SCENARIO_SAMPLING_OVERRIDES`
- `METHODS`

## 결과

결과는 `Evaluation/outputs/run_YYYYMMDD_HHMMSS/` 아래에 저장됩니다.

- `policy_comparison_results.csv`
- `stationary_results.csv`
- `nonstationary_results.csv`
- `boxplot_stationary_total_cost.png`
- `boxplot_nonstationary_total_cost.png`
- `boxplot_total_cost_all_methods.png`
- `summary_by_environment_method.csv`
