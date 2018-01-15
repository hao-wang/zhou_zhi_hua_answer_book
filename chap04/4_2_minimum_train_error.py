**使用最小训练误差作为决策树划分选择准则的缺陷。**

*Answer* The decision tree constructed via the "minimum training loss"
threshold is prone to the overfitting problem. That is, it can't generalize to
the test set, which is the goal of any machine learning algorithm.

For that matter, problem *4.1* is an extreme case, where we've proved you can
*always* get a 0 loss for training set with certain properties; in this case,
the decision tree has learnt everything about the training set, which is
guranteed to be overfitting.
