# TC-BERT: Large-Scale Language Model for Korean Technology Commercialization Documents
* Hyeji Hwang*, Changdae Oh*, Eunkyeong Lee, Taero Kim, Yewon Kim, Yunjeong Choim Sungjin Kim, Hosik Choi, and Kyungwoo Song, preprint 2022 (research project funded by [KISTI](https://www.kisti.re.kr/eng/))


## Organization
* `data`: location of datasets for document classification / recommendation
* `model`: location of the fine-tuned model checkpoint
* `pretrained/40ep_train9`: location of the pre-trained model checkpoint
* `vocab`: location of the vocabulary used for TC-BERT pre-training
* `doc_clf.py`: inference code for classification
* `doc_rec.py`: inference code for recommendtaion
* `ft.py`: fine-tuning code
* `doc_clf_notebook.ipynb`: inference code for classification (interactive)
* `doc_rec_notebook.ipynb`: inference code for recommendtaion (interactive)
* `run_ft.sh`, `run_rec.sh`: sample script for fine-tuning and recommendation

## Requirements
* Model checkpoints, datasets, and other resources used for the project are available through the MLAI@Yonsei Google Drive > Lab Meeting Slides > Individual Meetings > Changdae Oh > TC-BERT
* Data
  * `data/total_v2.1.csv` is required for fine-tuning and `data/etri_changdae_AFTER.csv` for the evaluation of fine-tuning
  * `data/inference_sampleset.csv` is required for the inference of document classification
  * `data/etri4rec_v2.csv` is required for the inference of document recommendation
* Model
  * `pretrained/40ep_train9/` should contain checkpoints from pre-trained TC-BERT
  * `model/` should contain fine-tuned model checkpoints, e.g., checkpoint-984 (best) or checkpoint-12300 (last), for classification and recommendation

## Document Classification
* you can run the classification via `python doc_clf.py` directly
* for interactive programming, check the `.ipynb` extension 
* after execution, you could find the prediction result as a `.csv` file in `result/` and get accuracy from the terminal.

## Document Recommendation
* run the command `sh run_rec.sh` by specifying source document id as `src_id` and other arguments.
  * For the first run, you should set the argument `run_scratch` as 1 to make a cache file of *TC-BERT embedding pool* for the entire evaluation documents.
  * After then, you could set `run_scratch` as 0 to load embedding pool from the cache
* likewise the case of classification you could also run this by `.ipynb` for interactive UI
* after execution, check the `result` directory to see the `.txt` files containing recommended docs from given source doc id.

## Fine-tuning for Document Classification
* see `ft.py`. you could try cross-entropy loss, [Dice loss](https://arxiv.org/abs/1707.03237v3), and [focal loss](https://arxiv.org/abs/1708.02002) for fine-tuning.
* refer to `ft_{method}_tune.sh` for hyperpararmeter tuning of each learning objectives
