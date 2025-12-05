### 1. Motivation and Overview

| **Title**            | **Description**                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset Name**     | CaraMLLo-Academic-Guardrail-Dataset.                                                                                            |
| **Creators**         | Anonymous                                                                                                                       |
| **Funding Source**   | Anonymous                                                                                                                       |
| **Target Task/Goal** | The specific purpose is to use this dataset to train context-specific guardrails, particularly for academic chatbot protection. |

---

### 2. Composition and Structure

| **Title**             | **Description**                                                                                                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Type**         | Text                                                                                                                                                                                                    |
| **Total Size**        | 4124 instances                                                                                                                                                                                          |
| **Languages**         | Brazilian Portuguese                                                                                                                                                                                    |
| **Text Unit**         | Each instance is a user prompt/question.                                                                                                                                                                |
| **Data Fields**       | - `id`: the instance's id<br>- `category`: instance's label<br>- `message`: the user prompt/question<br>- `explanation`: why is the message unsafe (used only for `unethical` and `off-topic` messages) |
| **Labels**            | `safe`, `unethical`, `off-topic`                                                                                                                                                                        |
| **Train subset**      | - Name: `train_fine_tuning_data_v2_restructured.json`<br>- Number of instances: 3626 <br>- `safe`: 1175 <br>- `unethical`: 1152 <br>- `off-topic`: 1299                                                 |
| **Validation subset** | - Name: `val_fine_tuning_data_v2_restructured.json`<br>- Number of instances: 389<br>- `safe`: 131<br>- `unethical`: 131<br>- `off-topic`: 127                                                          |
| **Test subset**       | - Name: `benchmark_data_v5.json`<br>- Number of instances: 109<br>- `safe`: 53<br>- `unethical`: 27<br>- `off-topic`: 29                                                                                |
| **Data Source**       | Synthetic generation.                                                                                                                                                                                   |


