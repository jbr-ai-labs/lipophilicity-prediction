#!/bin/bash
# starting number of experiment
i=10
for h in 50 200
do
	for f in 100 1000 10000
	do
		for p in 10 20
		do
			for c in 5 10
			do
				python train_proto.py -data logp_wo_aver -output_dir output/exp_$i -lr 5e-4 -n_epochs 100 -n_hidden $h -n_ffn_hidden $f -batch_size 16 -n_pc $p -pc_size 10 -pc_hidden $c -distance_metric wasserstein -separate_lr -lr_pc 5e-3 -opt_method emd -mult_num_atoms -nce_coef 0.01
				i=$(($i+1))
				echo $i
			done
		done

	done
done

