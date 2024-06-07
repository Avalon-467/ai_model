# ai_model
a repo collects different ai models<br>
人工智能模型集<br>
index:<br>
### base_model
customized_MLP: a mlp model which user can define the number of nodes in each layer of mlp freely and easily just by a list is composed of int numbers.<br>
通过整数列表便捷定义全连接层的各层节点数目。<br><br>
customized_conv: a multi_layers convolution model, which user can define it by a list of numbers<br>
通过整数列表便捷定义卷积层的各层节点数目<br><br>
reset_layer: partition and shuffer the feature map of conv<br>
分块打散特征图，以令卷积获得全局建模<br>

### train_test
trainer: train the specific model in specific trian_dataset<br>
通用训练器：指定模型和训练集<br><br>
classifier_tester: a tester for classifier task<br>
分类任务测试器




