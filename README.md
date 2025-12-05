# 5mins - 우리집 고양이도 메리 크리스마스 🎅

📢 2024년 2/겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다
🎉 2024년 2/겨울학기 AIKU Conference 열심히상 수상!


## 소개
우리집 고양이도 메리 크리스마스!

기존 image editing 모델들의 한계점으로 지적되어 온 것 중 하나는 객체 간의 관계성입니다. 
즉, 특정 객체만을 편집하거나 관계(두 객체의 위치, 크기 등)을 바꾸는 것에서 한계가 있었습니다. 
이번 프로젝트에서는 강아지, 고양이 사진을 크리스마스 분위기로 바꾸는 image editing을 하며 다음 문제를 해결해보고자 합니다. 

- 특정 객체를 추가하거나 수정하는 능력 높이기
- 다양성이 보장된 고품질의 image editing dataset 생성


## 방법론

### 데이터셋 생성

#### 1. 텍스트
   GPT-3.5 turbo 모델을 활용하여 input으로 주어진 instruction 문장을 패러프레이징하여 다양한 instruction 문장 생성
   ![image](https://github.com/user-attachments/assets/7fdac586-9710-46b0-ac69-ed7ea2a49cbf)

#### 2. 이미지
   위에서 생성한 instruction을 랜덤 샘플링하여 이에 맞는 edited image 생성, 이때 instruct pix2pix 모델을 사용하여 paired 데이터셋 생성
   다양한 생성 결과를 위해 image guidance scale을 하나의 값으로 고정하지 않고 여러 값을 사용

   ![image](https://github.com/user-attachments/assets/42cea74b-ae9e-4e93-bc1a-c8edd19c63bb)


### Fine-tuning
#### 0. 생성한 이미지 필터링
   생성된 이미지들을 manual-data-filtering을 통해 finetuning에 활용할 데이터를 선별

#### 1. 허깅페이스 삼중항 데이터셋 구축  
   만들어진 데이터셋을 선별하여 삼중항 데이터셋 구축 및 허깅페이스 업로드(input_image, instruction, output_image)
     
   ![image](https://github.com/user-attachments/assets/d18075e4-f6b1-439c-a872-7c91c843e43a)
  
#### 2. 학습 과정  
   diffusers에서 제공하는 train_instruct_pix2pix.py를 실행하여 further training 진행
   1390개 pair row에 대해서, 총 8000 step 만큼 파인튜닝 진행
     
   <img width="507" alt="image" src="https://github.com/user-attachments/assets/eafe17ab-e68e-4244-9644-ba1cd11cc9e0" />

#### 3. huggingface 배포
   - `pwnhyo/instruct-pix2pix-model`
     
   <img width="591" alt="image" src="https://github.com/user-attachments/assets/13bbf79a-959d-45c7-b4c4-db0a5753513f" />





## 환경 설정
```
conda create --name 5min python=3.9
conda actibate 5min

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .

cd examples/instruct_pix2pix
pip install -r requirements.txt

accelerate config (default)
```

## 사용 방법
### Trainining:
```
sh train.sh
```
### Inference:
```
python inference.py
```

## 예시 결과

![cat1](https://github.com/user-attachments/assets/c78a221b-c61e-4acd-bdf4-a3663b2cd557)
![dog1](https://github.com/user-attachments/assets/192009a2-c2fc-4ea2-8b63-6a8df86d6ea2)

