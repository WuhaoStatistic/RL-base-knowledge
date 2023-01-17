## Sarsa update
SARSA是TD的一个延申，这个名字的由来可以由下面这幅图诠释，Q是一个look-up table，记录了（S,A）的动作价值函数。
![1](/图片/8.png)
我们想要知道状态S下的动作A的value，那么我们就在状态**S**，进行动作**A**，收获一个**R**的同时到达下一个状态$\mathbf{S\prime}$,之后再采样一个动作$\mathbf{A\prime}$然后更新公式。注意上文中所有加粗的字母，连起来就是**SARSA**，注意到sarsa中的$Q(s\prime,a\prime)$是采样获得的是必须和环境交互的来的，所以他是on-policy。

算法流程如下图所示，注意到sarsa需要通过动作采样获得下一阶段的行为才行。如果我们用off-policy的办法，从我们的data set中获取，**我们无法保证我们的dataset在$S\prime$上是和我们的策略要求的分布相同（尤其是我们经常会在训练的前期后期调整$\epsilon$）**，也就无法做到SARSA算法。
![1](/图片/9.png)

Sarsa也同样可以采用n-step的思路去做
![1](/图片/10.png)
然后自然的引出Sarsa($\lambda$)
![1](/图片/15.png)
和之前介绍都一样，forward-view要求我们必须完成一整个trajectory才能更新我们的Q-table，否则我们无法获取到terminal的值，也就无法计算$q_t^\lambda$。
![1](/图片/16.png)
正经的算法也是基于backward-view来做的
![1](/图片/17.png)
总结一下到目前为止的一些东西
![1](/图片/18.png)
![1](/图片/19  .png)