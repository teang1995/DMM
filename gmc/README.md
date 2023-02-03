# Dynamic Mixed Margin for GMC

See `./gmc_code/supervised_dmm` where is the code corresponding to the Section 5.2., the bulk of code is borrowed from the [official implementation](https://github.com/miguelsvasco/gmc) of
"Geometric Multimodal Contrastive Representation Learning", ICML 2022.

So, please refer the above official repository for details about environment setup, installation, and downloading datasets.

Here, we only provide discriptions to reproduce our experiments (Table 4.) in the paper. In the Table 4. we trained the methods MulT, GMC, GMC with DMM on CMU-MOSEI train set, and evaluated them in the seven environments accross the observed modalities (Text, Vision, Audio).

---
## For reproducing MulT and GMC
```bash
sh mosei_baseline.sh $flag
```
You can run the five repeated train-eval experiments of MulT or GMC with above script, if `flag` is 0 the model will be MulT and 1 for GMC.



## For reproducing GMC with DMM
```bash
sh mosei_dmm.sh
```
You can run the five repeated train-eval experiments of GMC+DMM with above script.