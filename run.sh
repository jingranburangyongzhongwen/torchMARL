nohup python -u main.py --max_steps=1000000 --epsilon_anneal_steps=20000 --gpu='1' --optim='adam' --num=5> qmix-adam1.out  2>&1 &
sleep 10
nohup python -u main.py --max_steps=1000000 --epsilon_anneal_steps=20000 --gpu='0' --optim='adam' --num=5 --her=3> qmix-adam2.out  2>&1 &