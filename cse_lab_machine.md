# Query BERT Service on CSE Mahines

**First, you should active your ISTC account for CSE Lab** through https://cssystem.cse.ust.hk/UGuides/activation.html. 

About the deparment computing resoruces, please refer to [for PG](https://cssystem.cse.ust.hk/UGuides/PG/hkust_only/intro.html) or [for UG](https://cssystem.cse.ust.hk/UGuides/UG/hkust_only/intro.html).

We have deployed the BERT server on the CSD machine, you can login to `csl2wkxx` (xx is a number), e.g.,

```bash
ssh yourID@csl2wk53.cse.ust.hk
```

After login, type the following commands
```bash
bash
source /csproject/comp5211/2019Fall/source.sh
conda activate venv
```

Ideally, if you saw a `(venv)` in the beginning of your terminal line, everything 
is ready. Now you can try to query the server through BERT-serving client API.

```python
from bert_serving.client import BertClient
bc = BertClient(ip="lgpu1", port=10086, port_out=10087)
bc.encode(['What a nice day', 'everyone gets an A'])
```

The return will be 
```python
array([[-0.0155578 , -0.23251502,  0.178801  , ..., -0.16294768,
         0.09539697, -0.09102433],
       [-0.25249138,  0.10362521, -0.08484162, ..., -0.14833207,
         0.8393693 ,  0.50950617]], dtype=float32)
```


**Note, the `csl2wk` computers will restart at 7:00 am every day.** The hard-device
quota is very limited** (200MB) by the CSD**. Due to the limited computing resources, 
BERT-server may get stuck if there are too many queries simultaneously.