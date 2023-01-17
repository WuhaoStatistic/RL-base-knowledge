## 马尔可夫决策过程

强化学习作为一个概率模型，我们需要表示在某一个状态下采取某一个动作的概率。而站在真实的角度观察，进入下一个状态的概率还和上一个状态（毕竟上一个状态如果不进入当前状态，那么进入下一个状态的概率也会变化），上上一个状态。。。。。这样子的假设会导致问题变得无法解决。

在本章中，我们称state为状态，称action为动作，称reward为奖励，在不同的语境下，英文表示可能会有所不同，如在控制系统中，我们有另一套英文单词表达同样的对象。

因此，我们引入了一个马尔可夫性，也就是 当前状态的转移概率$s\prime$ 只和当前状态s有关。用数学语言表达为一个条件概率，条件概率中conditional on的部分只包含了当前的状态s和动作a。
$$
P_{ss\prime}^a = \mathbb{p}(S_{t+1}=s\prime,R_{t+1}=r|S_t=s,A_t=a)
$$
`之所以将t时刻的奖励写为`$R_{t+1}$`是因为在我们的运算过程中，获取奖励和进入下一个状态是同时发生的，在时刻上他们是同步的。`
除此之外，我们还认为对于策略而言，在每一个状态下采取动作的概率也只和当前所处状态有关
$$
\pi(a|s)=P(A_t=a|S_t=s)
$$
对于**状态价值函数**，也是一样的。价值函数中的Gt表示从当前状态开始一直到最终结束时所有奖励（未来奖励会衰减）的期望总和。

$$
\begin{aligned}
\mathcal{v}_\pi(s) &= \mathbb{E}_\pi(G_t|S_t=s)  \\
&= \mathbb{E}_\pi(R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+....|S_t=s)
\end{aligned}
$$


在状态价值函数中，我们并没有考虑动作带来的影响，所以我们在这里引入**动作价值函数**
$$
\begin{aligned}
q_\pi(s,a) &= \mathbb{E}(G_t|S_t=s,A_t=a) \\
&=\mathbb{E}(R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+....|S_t=s,A_t=a)
\end{aligned}
$$

### 贝尔曼方程 bellman equation
将状态价值函数带入动作价值函数，我们可以得到**贝尔曼方程**
首先推导出状态价值函数的贝尔曼方程
$$
\begin{aligned}
\mathcal{v}_\pi(s) &= \mathbb{E}(G_t|S_t=s,A_t=a) \\
&=\mathbb{E}(R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+....|S_t=sa) \\
&=\mathbb{E}(R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+3}+....)|S_t=s) \\
&=\mathbb{E}(R_{t+1}+\gamma \mathcal{v}_\pi(S_{t+1})|S_t=s)
\end{aligned}
$$
照葫芦画瓢，我们可以得到动作价值函数的贝尔曼方程
$$
\begin{aligned}
q_\pi(s,a) &= \mathbb{E}(G_t|S_t=s,A_t=a) \\
&=\mathbb{E}(R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+3}+....)|S_t=s,A_t=a) \\
&=\mathbb{E}(R_{t+1}+\gamma \mathcal{q}_\pi(S_{t+1},A_{t+1})|S_t=s,A_t=a)
\end{aligned}
$$
### 状态价值函数与动作价值函数的转换

根据他们的定义，我们可以写出这二者之间的转换关系。回忆一下状态价值函数表示处在某一状态时，之后可能会获得的所有奖励的期望。那么这时候我们有该状态下的动作价值函数以后，只需要将动作价值函数乘以对应发生的概率再求和即可。
$$
\mathcal{v}_\pi(s) = \sum_{a \in A}\pi(a|s)q_\pi(s,a)
$$
反过来，某一个时刻采取某一动作的动作价值函数也可以表示为
`采取该动作的奖励`+$\gamma$*`下一个状态的状态价值函数`。

注意采取一个动作可能并不一定能够确切的进入一个状态，比如在斑马线采取过马路的动作，可能被创死，也有可能安全度过。对于每一个状态，我们有在文章开头定义的$P_{ss\prime}^a$来定义。所以最终结果中的`下一个状态的状态价值函数`应该是一个sum的形式给出
$$
q_\pi(s,a)=R_s^a+\gamma\sum_{s\prime,r\prime}P_{ss\prime}^a \mathcal{v}_\pi
(s\prime)
$$
将两个式子结合起来，
$$
\begin{aligned}
\mathcal{v}_\pi(s) &= \sum_{a \in A}\pi(a|s)q_\pi(s,a) \\
q_\pi(s,a)&=R_s^a+\gamma\sum_{s\prime,r\prime }P_{ss\prime}^a \mathcal{v}_\pi
(s\prime)
\end{aligned}
$$
我们能够显式的推导出状态价值函数和动作价值函数,这个过程就是分别将一条式子带入另一条式子，只需要注意时刻问题，弄清楚哪些地方用现在的时刻哪些地方是将来的时刻。下式就是著名的**贝尔曼期望方程**的显式表达
$$
\begin{aligned}
\mathcal{v}_\pi(s) &= \sum_{a \in A}\pi(a|s)(R_s^a+\gamma\sum_{s\prime,r\prime}P_{ss\prime}^a \mathcal{v}_\pi
(s\prime)) \\ 
q_\pi(s,a)&=R_s^a+\gamma\sum_{s\prime,r\prime }P_{ss\prime}^a\sum_{a\prime \in A}\pi(a\prime|s\prime)q_\pi(s\prime,a\prime)
\end{aligned}
$$

### 最优价值函数
这里并没有刻意强调是**最优状态价值函数**还是**最优动作价值函数**，因为他们二者可以互相推导。我们的目的是通过训练学得一个最优的policy$\pi^\star$，一般来说强化学习问题很难直接学到一个最优策略，大多数情况下是通过比较若干策略来获得一个相对优解，也就是我们常说的**local optimal**。

最优状态价值函数的定义：
$$
\mathcal{v}^\star(s)= \underset{a}{max} \ \mathcal{v}_\pi(s)
$$
最优动作价值函数的定义
$$
q^\star(s,a)=\underset{\pi}{max}\ q_\pi(s,a)
$$
对于最优解而言，我们可以定义策略：

$$ \pi^\star(a|s)=\left\{
\begin{matrix}
 1\ ,if\  a=argmax_{a\in A} q^\star(s,a) \\
 0\ ,else 
\end{matrix}
\right.
$$
将上面式子带入之前得到的递推式
$$
q_\star(s,a)=R_s^a+\gamma \sum_{s\prime \in S}P_{ss\prime}^a \ \underset{a\prime}{max}q_\star(s \prime,a \prime)
$$
$$
v_\star(s)=\underset{a}{max}(R_s^a+\gamma \sum_{s\prime \in S}P_{ss\prime}^a \mathbb{v}_\star(s\prime))
$$
对比一下式子
$$
\begin{aligned}
\mathcal{v}_\pi(s) &= \sum_{a \in A}\pi(a|s)(R_s^a+\gamma\sum_{s\prime,r\prime}P_{ss\prime}^a \mathcal{v}_\pi
(s\prime)) \\ 
q_\pi(s,a)&=R_s^a+\gamma\sum_{s\prime,r\prime }P_{ss\prime}^a\sum_{a\prime \in A}\pi(a\prime|s\prime)q_\pi(s\prime,a\prime)
\end{aligned}
$$
最优动作价值函数和普通动作价值函数，因为我们确定了最优的动作，所以不存在对于所有动作加权平均，只寻求下一个状态中动作价值最大的动作。
对于状态价值函数，我们也不再对当前状态的所有动作求加权平均，而是选择一个动作价值最大的动作。
### 例子
下面这个图 能非常好的解释上述理论是如何工作的
![1](/图片/1.png)

我们假设从左上到右下的状态分别为**S1,S2,S3,S4**。
注意最底下的黑点并不是一个状态，他表示S4进行pub动作之后有3种后续的可能状态，进入每一状态的概率就是我们之前提到的$P_{ss\prime}^a$，这里实际上还是一种简化形式，因为奖励R实际上也是一个随机变量。在我们这个例子中，所有的奖励都是固定的。

### 1 基于状态价值函数的解法。
$$
v_\star(s)=\underset{a}{max}(R_s^a+\gamma \sum_{s\prime \in S}P_{ss\prime}^a \mathbb{v}_\star(s\prime))
$$
![2](/图片/2.jpg)

### 2 基于动作价值函数的解法
在这种解法中，首先看最终结束状态，s4的时候学习的动作可以直接得到最优动作价值函数为10
$$
q_\star(s,a)=R_s^a+\gamma \sum_{s\prime \in S}P_{ss\prime}^a \ \underset{a\prime}{max}q_\star(s \prime,a \prime)
$$
![3](/图片/3.png)