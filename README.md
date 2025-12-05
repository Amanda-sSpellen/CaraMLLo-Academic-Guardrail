# CaraMLLo-Academic-Guardrail
This repository contains the code, data, and fine-tuned model for CaraMLLo, a domain-specific guardrail module designed to responsibly query Brazil's leading academic platform, Currículo Lattes. 

We present a fine-tuned version of the Llama Guard 3 1B model, enhanced with a custom taxonomy to identify academically safe, unethical, and off-topic content. Our research shows that this domain-specific approach significantly improves classification accuracy, highlighting the importance of adaptation for responsible AI use. This repository includes the PEFT weights for CaraMLLo, the full training and evaluation code, and the synthetic dataset used to fine-tune the model, along with the code used for its generation and a comprehensive datasheet.

---
## Abstract
Large Language Models (LLMs) have recently gained significant popularity and have demonstrated broad applicability across various tasks, including chatbots and conversational AI. However, without proper safeguards, such systems risk enabling biased, inappropriate, or unethical queries. We present Cara**MLL**o, a domain-specific guardrail module for querying _Currículo Lattes_, Brazil’s leading academic platform. To overcome the limits of generalist moderation tools, we fine-tuned the Llama Guard 3 1B model using a synthetic dataset based on a custom taxonomy: academically safe, unethical, and off-topic content. The fine-tuned model achieved a 26.40 percentage point improvement in accuracy over the baseline, showing strong classification performance, especially for unethical content. The results underscore the value of domain adaptation for responsible AI use.

---
## Repository structure

```
.
├─ README.md
├─ dataset
│  ├─ Datasheet-CaraMLLo-Academic-Guardrail-Dataset.md
│  ├─ benchmark_data_v5.json  // test subset
│  ├─ guardrail_categories_v1.json  // category definitions 
│  ├─ train_fine_tuning_data_v2_restructured.json  // train subset
│  └─ val_fine_tuning_data_v2_restructured.json  // validation subset
├─ eval_script
│  └─ generate_10_tests.py  // creates average confusion matrices for binary and multiclass classification
├─ model_weights
│  └─ lg_ft_3k_v15_pt1_2.zip
└─ train_script
   ├─ get_academic_unsafe_dataset.py  // custom function for training
   └─ lg_ft_3k_v15_pt1.sh  // training script
```


---
## Citation

TODO
