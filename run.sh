nohup python -u main.py --map='5m_vs_6m' --alg='qmix'   --gpu='a' > qmix.out   2>&1 &
sleep 10
nohup python -u main.py --map='5m_vs_6m' --alg='cwqmix' --gpu='a' > cwqmix.out 2>&1 &
sleep 10
nohup python -u main.py --map='5m_vs_6m' --alg='owqmix' --gpu='a' > owqmix.out 2>&1 &
