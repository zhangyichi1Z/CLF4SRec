import os

sim = ['cos']
lmd = [0.1, 0.3, 0.5, 0.7, 0.9]
drop = [0.1, 0.3, 0.5, 0.7, 0.9]
dataset = ["ml-1m", "Amazon_Beauty", "Amazon_Clothing_Shoes_and_Jewelry", "Amazon_Sports_and_Outdoors"]
lmd_tf = [0.1, 0.3, 0.5, 0.7, 0.9]
for data in dataset:
    for l in lmd:
        for s in sim:
            for d in drop:
                for l_tf in lmd_tf:
                    os.system("python run_seq.py --dataset={} --train_batch_size=256 "
                              "--lmd={} --lmd_sem=0.1 --model='CLF4SRec' --contrast='us_x' --sim={} "
                              "--tau=1 --hidden_dropout_prob={} --attn_dropout_prob={} --lmd_tf={}".format(data,
                                                                                               l,
                                                                                               s,
                                                                                               d,
                                                                                               d,
                                                                                               l_tf
                                                                                               ))
