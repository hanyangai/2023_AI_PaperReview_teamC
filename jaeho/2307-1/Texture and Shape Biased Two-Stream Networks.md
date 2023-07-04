# Texture and Shape Biased Two-Stream Networks for Clothing Classification and Attribute Recognition

[Paper Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Texture_and_Shape_Biased_Two-Stream_Networks_for_Clothing_Classification_and_CVPR_2020_paper.pdf)
<br><br>

<aside>
🔥 논문의 내용을 참고했을 뿐, Related works나 제목과 그에 대한 내용들은 논문의 실제 내용들과는 다를 수 있습니다.
<br><br>
</aside>

# Background

먼저 이 당시에 근무하고 있던 회사 업무의 연장선으로 선택했던 논문이고, 당시에 Fashion 이미지에 대해서 Attributes들을 Classification 하는 모델을 담당했었다.

단순한 Multi class Classification일 수도 있지만 Fashion 이미지의 특성 상 분류해야 하는 속성의 경우 Multi Class보다는 Multi Label Classification에 조금 더 가깝다.
"C:\Users\user\Documents\2023_AI_PaperReview_teamC\jaeho\2307-1\Texture and Shape Biased Two-Stream Networks"
![[https://www.analyticsvidhya.com/blog/2021/07/demystifying-the-difference-between-multi-class-and-multi-label-classification-problem-statements-in-deep-learning/](https://www.analyticsvidhya.com/blog/2021/07/demystifying-the-difference-between-multi-class-and-multi-label-classification-problem-statements-in-deep-learning/)](Texture%20and%20Shape%20Biased%20Two-Stream%20Networks/Untitled.png)

[https://www.analyticsvidhya.com/blog/2021/07/demystifying-the-difference-between-multi-class-and-multi-label-classification-problem-statements-in-deep-learning/](https://www.analyticsvidhya.com/blog/2021/07/demystifying-the-difference-between-multi-class-and-multi-label-classification-problem-statements-in-deep-learning/)

그러던 중에 CVPR 2020에 억셉된 논문들 중에서 이 논문을 알게 되었고, 논문의 구조 자체도 그리 어렵지 않아서 졸업 논문의 Baseline으로 채택을 했다.

# Introduction

- 기존의 ImageNet Pretrained 모델들의 경우, Shape 정보보다는 Texture 정보에 편향되어 있음
    
    ![[https://arxiv.org/pdf/1811.12231.pdf](https://arxiv.org/pdf/1811.12231.pdf)](Texture%20and%20Shape%20Biased%20Two-Stream%20Networks/Untitled%201.png)
    
    [IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE; INCREASING SHAPE BIAS IMPROVES ACCURACY AND ROBUSTNESS](https://arxiv.org/pdf/1811.12231.pdf)
    
- Fashion 이미지는 Texture 정보 뿐만 아니라 Shape에 대한 정보도 풍부하기 때문에 기존에 ImageNet Pretrained 모델의 경우 정보를 제대로 활용하지 못하고 있는 점을 지적
- 논문의 제목에서 나와있듯이, Texture와 Shape에 대한 각각의 Stream이 존재하고 두 가지의 정보들을 Joint Learning을 통해서 기존의 Attribute Recognition의 성능을 더 올렸다고 주장

# Related Work

- Multi-Class(Single Label) Classification `vs` Multi-Label Classification
    
    Multi-Class Classification이라면 주어진 하나의 이미지에 대해서 하나의 Label이 대응되는 경우로 흔히들 많이 알고 있는 Classification 태스크이다.  하지만 실생활의 이미지의 경우, 특히 Fashion이미지로 예를 들어보자면 매우 다양한 정보들을 갖고 있다.
    
    ![Untitled](Texture%20and%20Shape%20Biased%20Two-Stream%20Networks/Untitled%202.png)
    
    이 이미지로 예시를 들어보면 단순히 `티셔츠` 혹은 `상의` 와 같이 간단히 태깅을 할 수도 있을 것이다. 하지만 이 이미지를 세분화해서 Label을 생각해 본다면 다음과 같이 표현할 수도 있다.
    
    - Single Label Attribute
        - neckline : round neck
        - kind_of_sleeve : setin sleeve
        - length_of_sleeve : short
        - fit : normal
        - pattern : stripe
        - fabric : cotton
    - Multi Label Attribute
        - color : white, black
    
    *추가적으로 이처럼 하나의 이미지에서 좀 더 다양한 분류를 Fine-grained Classification이라고도 부른다.*
    
    일반적으로 분류코드들을 보면 one-hot 인코딩 형태로 다음과 같이 표시한다.
    
    색깔을 예로 들어 `{Red, Green, Blue, Black, White}`의 라벨들이 있다고 해보자.
    
    만약 입력으로 들어오는 이미지가 Black 단색이라면  `[0, 0, 0, 1, 0]` 로 4번째 Label에 대해서 표현 할 수 있고, 위 예시 이미지처럼 Black과 White가 함께 존재하는 Stripe 패턴을 갖는 옷이라면 `[0, 0, 0, 1, 1]` 로 표현할 수 있다.
    
    데이터의 Label은 위처럼 간단하게 표현할 수 있다.
    
    그리고 Multi Class와 Multi Label의 차이점 중 코드적인 측면에서 가장 큰 차이는 Multi Class의 경우 `Softmax`를 사용하고, 그 중 값이 가장 큰 하나의 라벨만 예측하면 되는 것에 반해, Multi Label은 조금 다르다.
    
    Multi Label은 5개의 라벨에 대해서 각각 Binary Classification을 한다고 생각을 하면 좀 더 이해하기 쉽다.
    이미지에서 Red가 존재하는가 아닌가, Green이 존재하는가 아닌가와 같은 방식이다. 이것은 단순하게 `Softmax` 대신에 `Sigmoid`를 사용해서 구현하면된다.
    
    ```python
    import torch
    import torch.nn as nn
    
    class Model(nn.Module):
    	def __init__(self):
    		super(Model, self).__init__()
    		self.conv = nn.Conv2d(...)
    	
    	def forward(self, x):
        out = self.conv(x)
    		out = torch.sigmoid(out)
    		return out
    ```
    
    이때 결과를 예로 들어보자면
    
    `[0.0001, 0.0003, 0.0002, 0.8, 0.6]` 이렇게 표현할 수도 있다.
    위에 나온 수치들은 Sigmoid값이기에 `0.8`이 **80퍼센트의 확률로 검은색이 존재한다** 와는 다른 의미이다. 다만 높은 확률로 입력 이미지에는 검은색이 존재한다고 판단하면 되는 것이고, 하얀색도 다른 라벨에 비해서는 다소 높은 확률로 존재한다고 표현할 수 있다.
    이후에는 결과를 확인하는 방법은 상황에 따라 다르다.
    
    `Accuracy`를 주로 사용하는 일반 Classification과는 다르게  Multi Label 에서는 상황에 따라서 `Recall`이나 `Precision` 혹은 `F1-Score`등 다양한 메트릭을 활용한다.
    
    이때 Top-k 메트릭을 사용할 수도 있지만, 그렇지 않을 경우 이 또한 실제 서비스나 상황에 따라서 Threshold를 정해서 해당 이미지에서 어떤 색이 존재하는지를 판단해야 하는 경우도 존재한다.
    
    예를 들어 Threeshold로 `0.7`의 값을 사용한다면 입력 이미지에 대한 예측값은 0.8의 값을 갖는 검은색만 있다고 판별하는 것이고, 보다 낮은 하얀색(`0.6`)은 없다고 판별하는 것이다. 그렇기 때문에 이 Threshold로 인해서 `Recall`, `Precision`, `F1-Score`와 같은 메트릭의 값또한 유동적으로 변할 수 있고, 그렇기에 실무에서 원하는 Task에 맞춰 적절한 Metric과 적절한 Threshold를 찾기위한 최적화가 필요할 수도 있다. 
    
- Multi Task Learning(Multiple Loss)
    
    이 논문은 여러가지 Loss를 활용하고 있고, 여러가지 Task에 대해서 모델을 학습시키고 있다. 이러한 Task를 Multi Task Learning이라고도 부른다. 뒤에서 설명할 예정이지만 이 논문에서는 두 개의 Stream에 대해서 각기 다른 Task를 학습시키고 있다.
    
    하나의 Stream에서는 Category Classification과 Attribute Recognition, 나머지 다른 하나의 Stream에서는 Landmark Detection을 Task를 담당하고 있고 이때 Loss는 총 4개를 사용해서 학습하고 있다.
    
    한 번의 학습으로 총 3개의 결과를 예측할 수 있는 모델은 현재 실생활의 분류에서 매우 유용할 수 있다. Task마다 각각 모델이 존재해야 한다면 Category, Attribute, Landmark 단일 모델로 구성을 하게되고, 그렇게 되면 하드웨어 리소스도 그만큼 많이 필요할 것이다. 
    
    하지만 Multi Task Learning의 경우, 하나의 Backbone을 이용해 위처럼 여러 태스크를 해낸다면 리소스적인 측면에서도 절약이 가능하다. 그리고 일반적으로 Multi Task Learnig의 장점으로는 동일한 입력에 대해서 A Task에 대한 정보를 이용할 뿐만 아니라 B Task에 대한 정보(지식)도 함께 학습하면서 성능을 올리기 때문에 이러한 지식의 공유로 개별 Task의 측면에서도 좋다고 말을 하기는 한다.
    
    하지만 이는 실제로 너무 이상적이지 않아 싶을 정도로 학습을 실제로 시켜본다면 학습이 잘 진행되지 않고, 고려해야 할 사항도 너무 많다. Task마다 궁합이 잘 맞는 Task들이 존재할 수도 있고, 여러 Loss를 사용할 경우 각 Loss에 대해서 특정 Loss에는 가중치를 더 준다던가, 그 균형을 맞추기는 쉽지 않다.
    

# Dataset

- DeepFashion: Attribute Prediction
    
    ![Untitled](Texture%20and%20Shape%20Biased%20Two-Stream%20Networks/Untitled%203.png)
    
    [Category and Attribute Prediction Benchmark](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)
    
- 289,222장의 의류 이미지
- 50개의 category 정보, 1000개의 attribute 정보
- Bbox annotation, Landmark annotation
<br><br>
# Architecture

![Untitled](Texture%20and%20Shape%20Biased%20Two-Stream%20Networks/Untitled%204.png)

논문에서 제안하는 모델의 구조는 위와 같다.

Texture Biased Stream과 Shape Biased Stream, 두 개의 Stream이 존재하고 Texture Stream에서는 Category Classification(Multi Class), Attribute Recognition(Multi Label), Shape Stream에서는 Landmark Detection을 수행한다.

이때 사용되는 Backbone은 VGG를 사용하고 있고, conv5 이후에 각 Stream의 Backbone에서 나오는 Feature를 Concat시켜서 Category와 Attribute에 대해서 예측한다.

이때 앞서 Introduction에서 얘기 했듯이 ImageNet Pretrained Weight는 Texture에 편향되었다는 연구도 존재했기에 Shape Stream에서는 VGG를 Scratch부터 학습 시킨다.

Lanrmark Detection Module은 다음과 같이 구성되어 있다.

![Untitled](Texture%20and%20Shape%20Biased%20Two-Stream%20Networks/Untitled%205.png)

Landmark Detection에서는 두 가지를 예측하는데 하나는 Landmark Location에 대한 예측과 Landmark가 존재하는지 안하는지를 판단하는 Visibility에 대해서 예측한다.

이때 Landmark Visibilty역시 각 Landmark들이 존재 하는지, 안하는지를 판단하기 때문에 Multi Label과 같은 Sigmoid를 사용하고 있고, Landmark Location은 각 채널 별로 알맞은 Landmark(총 8개의 포인트)를 예측하기 위해서 총 8개의 Channel을 갖는 Landmark 히트맵을 예측한다.
<br>
<br>
# Training

Scratch부터 학습시키는 Shape Stream의 때문에 첫 3에폭정도는 Shape Stream에 대해서만 높은 Learning Rate로 선행학습을 시키고, 이후에 LR을 낮춰서 모델 전체를 학습시킨다.

Loss는 모두 4개로 Category Loss는 CE(Cross Entropy Loss), Attribute Loss는 BCE(BInary Cross Entropy Loss)를 사용한다. Landmark Visibility 역시 Attribute Loss와 같이 BCE Loss, 그리고 Landmark Location에 대해서는 MSE Loss를 사용한다. 하지만 이때 이미지별로 보이는 Landmark와 보이지 않는 Landmark들이 존재하기에 다음과 같이 MSE loss를 살짝 변형해서 사용한다.

$$
L_{landmark} = \sum^K_{k=1}v_k^{GT}\sum_{x,y}||S_k{(x,y)}-S_k^{GT}(x,y)||_2
$$

- $S^{GT}_k$ : landmark 히트맵의 GT
- $S_k$ : 예측한 히트맵
- $v^{GT}_k$ : k번째 포인트의 visibility에 대한 GT
<br>
<br>
# Experiment

![Untitled](Texture%20and%20Shape%20Biased%20Two-Stream%20Networks/Untitled%206.png)

그렇게 학습시킨 논문의 결과는 위와 같다고 한다.

Pre-trained 항목은 ImageNet Pretrained를 의미하고 이는 Texture, Fabric을 나타내는 Attribute들에서 강세를 보이고 있다. 반면에 Joint Learning을 통해서 Shape 정보를 함께 사용해서 학습시킨 성능을 보면 Texture와 Fabric에 대해서 수치는 다소 떨어지기도 했지만 Shape와 Part의 성능이 실제로 상승한 것을 확인 할 수 있다.

그에 더해서 Category역시 Shape 정보가 추가됨으로써 성능이 올랐고, 이러한 점들을 통해서 기존에 Texture 정보에 편향되어 있는 모델에 Shape 정보를 더해서 기존의 분류 성능을 올린 점에 대해서 이야기하고 있다.
<br>
<br>
# 마치며…

논문을 보다보면 모델의 구조도 단순하고, 특별한 모듈도 없고, 왜 VGG를 사용했을까 하는 생각이 들면서 개선해보거나 다른 실험들도 가능 할 것 같은 느낌이었다. 하지만 일단 여러가지 문제들이 존재했다.

일단 오픈되어 있는 코드가 없다. CVPR이라면 그래도 메이저 학회이고 여기에 억셉이 된 논문이라면 자랑스럽게 공개를 할만한데 아무리봐도 없다. 그렇기에 처음부터 직접 다 구현을 해야했다.

추가적으로 정리를 하며 생략한 실험이나 정보들도 많이 존재하지만 무엇보다 구현을 하면서 느끼는 것은, 실험에 대한 상세한 Implementation 역시 제공하지 않고 있다. 하다못해 데이터를 준비할때, DeepFashion 데이터에서 Landmark에 대한 학습데이터를 만들때, 8개의 Landmark 포인트에 대해서 Heatmap을 만들어야 하는데, 단순히 생각하면 까만 도화지에 Landmark Point들이 하얀색 점으로 찍혀있는 것이다. 하지만 이때 학습을 좀 더 용이하게 하기 위해서 각 포인트에 대해서 Gausian을 적용하는데, 적용하고 나면 하얀점은 중심 포인트에 대해서 서서히 까맣게 변하는 형태를 띄게 된다. 이때 사용되는 Gausian 분포에 대한 자세한 implementation에 대해서도 제공하지 않는다.

이처럼 데이터 측면에서도 사실 DeepFashion이 그렇게 퀄리티가 막 좋은 데이터는 아니기에 일반적으로 논문과 코드를 공개하면 데이터나 모델 weight도 공개할만한데 아무 공개된 정보가 전혀 없었다.

실제 구현을 했을때도 논문에서 보여주는 수치들과는 너무 거리가 있는 수치들이었고, 따라잡을 수 없었다.

그리고 Loss를 구현할 때도, 논문에도 나와있지만 4개의 Loss가 존재하는데 학습할 때, Attribute Loss에 대해서만 가중치를 500이나 주고 학습을 했다.

$L_{category}$, $L_{attribute}$, $L_{visibility}$, $L_{landmark}$의 가중치 = `1:500:1:1`

Multi Loss를 학습시키는 경우, 특정 Loss에 대해서 가중치를 주는 경우는 많이 존재하지만 이렇게 500이라는 큰 값을 주는 경우는 아무래도 흔치 않았다.

하지만 이에 대한 설명은 딱히 존재하지 않는다. 그래서 더욱 답답한 점이다.

위와 같은 내용들을 보면 왜 공개를 안했는지도 조금 이해가 되기 시작했다. 하지만 이러한 부분들에서 개선을 해볼 수도 있는 것이고 이 논문을 베이스라인으로 잡고 논문을 작성중인 상황이다.