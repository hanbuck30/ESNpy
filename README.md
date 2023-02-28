# ESNpy
ESN(Echo-State-Network)
* 1. 전체적으로 ESN을 학습에 있어서는 먼저 ESN에 알맞은 parameters를 입력하고 정의한다
* 2. train values를 esn모델에 learning을 통해 input w, W를 learning 하고 fix시킨다
* 3. train values의 learning된 weight들과 구하고자 하는 output을 linear regression에 fit 한다
* 3. test values를 esn모델에 learning 하지말고 통과 시킨다
* 4. test values의 learning된 weight들을 linear regression에 predict를 통해 구하고자 하는 target을 알 수 있다
* # 참고사항으로 esn에서 fit은 input 길이에 제한이 없지만 linear regression을 할 때에는 input의 길이에서 initLen을 뺀 길이 만큼을 피팅한다
