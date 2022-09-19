# Collaborative-Filtering-Based-Recommender-System
My task response to Huawei.


Depended Libraries:
<br />
numpy<br />
pandas<br />
os<br />
tensorflow<br />
tensorflow: keras<br />
tensorflow.keras: layers<br />
pathlib: Path<br />
matplotlib.pyplot<br />
sklearn: preprocessing<br />
tensorflow.keras.models: Sequential<br />
tensorflow.keras.layers: Flatten,Embedding,Dense<br />
sklearn.model_selection: train_test_split<br />


![0 01_LR_Ressult](https://user-images.githubusercontent.com/76007933/191117490-a3afe9b3-cf13-4f00-8ea2-0de4c99f3be7.png)
<br />
with 0.01 Learning rate and "he_normal" embedded initilizer, that loss results acquired as can be seen at graph.


![0 01_LR_Uniform](https://user-images.githubusercontent.com/76007933/191117739-ff335b80-0d4a-47a2-bab0-32477d63b27f.png)
<br />
with 0.01 Learning rate and "RandomUniform" embedded initilizer, that loss results acquired as can be seen at graph.

![0 01_LR_lecun](https://user-images.githubusercontent.com/76007933/191117852-80c32d53-f428-4f55-8dc4-f45b31fcfaf0.png)
<br />
with 0.01 Learning rate and "lecun_uniform" embedded initilizer, that loss results acquired as can be seen at graph.
<br />
From my hyperparameter tunings, best results acquired from 0.01 Learning rate.




USAGE EXAMPLE:
<br />
![Ekran Görüntüsü (77)](https://user-images.githubusercontent.com/76007933/191122993-0875183f-2f51-464d-9f5a-6a2dce1415e9.png)<br />
As can be seen at the picture, program recommends users 5 different films that user never watched.
<br />
<br />
<br />
<br />
<br />
QUESTION 2 (Discuss deployable deficiency of the model at Question 1 in terms of industrial perspective. Suggest a proposal to improve it.):
<br />
This model can be used to recommend shopping app users better product. Also can be used to determine wich equipments, parts, tools or industrial products needs to get prioritized. Especially when we talk about Industry 5.0, this algorithms can help manufactories to run itself.<br />
<br />
** This model can be improved with much mor data, much more deeply examined hyperparameter tuning and deeper-complexier neural network.<br />
** Also for lightier algorithms, apriori like ML tools can be used since they are not needing that much process power.<br />
