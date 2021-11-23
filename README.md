# SETR_project

## Dataset <br />
 구글드라이브 공유된 데이터 -> oct_data\train\... & oct_data\test\... 로 이동 <br />
 
 | OCT machine | # patient | Train | Validation |
|-------------|-----------|-------|------------|
| Cirrus      | 1-24      | 6-24  | 1-5        |
| Spectralis  | 25-48     | 30-48 | 25-29      |
| Topcon      | 49-70     | 53-70 | 49-52      |

## Run
python SETR_train.py --model=="model_name"

**"model_name"** can be SETR_Naive_S, SETR_Naive_L, SETR_Naive_H, SETR_PUP_S,  SETR_PUP_L, SETR_PUP_H, SETR_MLA_S, SETR_MLA_L, SETR_MLA_H <br />
default = SETR_Naive_S

## Checkpoint

**best_model.pth** : validation loss가 가장 적은 모델의 파라미터<br />
**last_model.pth** : 마지막 epoch 이후 저장된 모델 파라미터<br />
test_ : validation 일부 저장

