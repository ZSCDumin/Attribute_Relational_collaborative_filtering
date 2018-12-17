<< comment
python NeuralFM.py --dataset frappe --hidden_factor 64 --layers [64] --keep_prob [0.8,0.5] --loss_type square_loss --activation relu --pretrain 0 --optimizer AdagradOptimizer --lr 0.05 --batch_norm 1 --verbose 1 --early_stop 1 --epoch 200
comment

python He_RFM.py \ 
   --dataset frappe \ 
   --hidden_factor 64 \ 
   --attention False \ 
   --keep_prob 0.5 \ 
   --pretrain 0 \ 
   --optimizer AdagradOptimizer \ 
   --lr 0.05 \ 
   --batch_norm 1 \ 
   --verbose 1 \ 
   --epoch 200

