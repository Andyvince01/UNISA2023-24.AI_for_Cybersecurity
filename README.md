# 🎭 UNISA23-24.AI_for_Cybersecurity

This project focuses on **assessing the resilience of a face recognition-based access control system** against adversarial attacks and investigating effective defensive strategies. The project involves several key stages, starting with the selection of **100 identities from the VGG-Face2 dataset’s training set** and the creation of a test set comprising at least **1000 images** (with a minimum of 10 images per identity). 

To run the project, execute the notebook named **`project.ipynb`**. Due to the large size of the generated attacks, totaling over 30GB, they were not uploaded. Therefore, to test the code, you will need to rerun the commands, ensuring to set the appropriate generation flags to True.

However, the data necessary for plotting SECs has been successfully loaded. Below are detailed explanations of the various Python modules included in the folder:

- **`project.ipynb`**: Notebook containing the entire project implementation.
- **`Datasets/VGG_Faces2`**: This directory contains the class responsible for organizing the data structure required by Senet50, as specified in the repository [github](https://github.com/cydonia999/VGGFace2-pytorch.git).
- **`Attacks/artattackwrapper`**: This module likely includes functionalities related to handling and executing adversarial attacks.
- **`Attacks/preprocessing`**: Module containing preprocessing techniques used to prepare data or mitigate adversarial effects.
- **`Attacks/carlini`**: Code for generating Carlini adversarial attacks, optimized by increasing the number of CPU threads for execution.
- **`Attacks/deepfool`**: Code for generating DeepFool adversarial attacks, similarly optimized for faster execution using multiple CPU threads.

The last two directories, `carlini` and `deepfool`, provide scripts specifically for generating these types of adversarial attacks. They are optimized for efficiency, particularly by leveraging multi-threading capabilities of the CPU. Running them outside the notebook environment has demonstrated significantly faster execution times. Nonetheless, the notebook sections include the necessary code lines to correctly execute these attacks.

## 📊 **Initial Phase: Accuracy Evaluation**

The initial phase evaluates the **accuracy of a face recognition network** (referred to as `NN1`) on this constructed test set. This phase is crucial for establishing a baseline performance measure against which the impact of adversarial attacks will be assessed.

## ⚔ **Adversarial Attacks**

Subsequently, **adversarial examples** will be generated using the `ART` library, targeting `NN1`. The impact of these adversarial examples will be assessed using **Security Evaluation Curves**. These curves will provide a visual representation of the system's vulnerability to adversarial attacks.

## 🔄 **Transferability Evaluation**

An additional classifier, trained on the VGG-Face2 dataset, will be chosen and evaluated on the "clean" test set to study the **transferability of adversarial examples** to this second classifier (`NN2`). This step is essential for understanding how adversarial examples affect different models and to what extent adversarial vulnerabilities are shared across classifiers.

## 🛡️ **Defense Mechanisms**

Finally, the project includes the **implementation and evaluation of at least one defense mechanism**. The focus will be on either detecting adversarial samples or employing pre-processing techniques to enhance the system's robustness. The effectiveness of these defense strategies will be measured by their ability to mitigate the impact of adversarial attacks and maintain the system's accuracy.
