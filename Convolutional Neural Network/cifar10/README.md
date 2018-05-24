# CIFAR 10


## Node 1
``` 
python main.py --lr 0.1 --world-size 2 --dist-url 'tcp://<master ip>:<port>' -r --epochs 100 --rank 0
```

## Node 2 
``` 
python main.py --lr 0.1 --world-size 2 --dist-url 'tcp://<master ip>:<port>' -r --epochs 100 --rank 1
```
