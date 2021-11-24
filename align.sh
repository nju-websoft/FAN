python align.py \
--output_dir ./output \
--device cuda:0 \
--pos_file ./output/mining/nyt10/pos.txt \
--unlabeled_file ./output/mining/nyt10/unlabeled.txt \
--rel2id ./data/nyt10/rel2id.json \
--epoch 10 \
--batch_size 4 \
--lr 1e-5 \
--pretrain ./bert/bert-base-uncased \
--optimizer sgd \
--contrastive \
--alpha 0.01 \
--beta 0.01 \
--gamma 0.0001 \
--tau 1.0 \
--weighting \
--max_bag_size -1 \
--freeze

