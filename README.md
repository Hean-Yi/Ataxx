## 这是一个开发日志，记录Ataxx的开发

### 准备工作：

这个Ataxx我想用神经网络辅助蒙特卡洛搜索来执行，并且在后续添加高性能计算模块进行辅助训练AI，所以目前需要学习的有：

- 神经网络（打算用卷积神经网络）、蒙特卡洛搜索树

---

```
以下是我找的一种训练方法，用AI解释了一下，我们可以仿照这个方法来训练AI

KataGo 是一个用于围棋的开源人工智能系统，基于深度学习和强化学习，类似于 AlphaGo 但比其更高效、灵活且强大。KataGo 的方法主要结合了 **深度卷积神经网络** 和 **蒙特卡洛树搜索**（MCTS）。它在 **AlphaGo Zero** 的基础上做了改进，并将其方法广泛应用到多个棋类，包括围棋和其他策略游戏。

KataGo 通过以下几个核心技术实现了其强大的性能：

### **1. 深度卷积神经网络（CNN）**

KataGo 使用深度卷积神经网络（CNN）来评估棋盘的局面。网络的主要作用是输出两个东西：

- **局面评估（Value）：** 这个值表示当前局面对于玩家（白方或黑方）的胜利概率。

- **行动策略（Policy）：** 这个值表示每个可能走子的概率。


#### 网络结构：

KataGo 的网络结构有一些变化，以适应不同的棋盘尺寸和复杂的游戏策略。网络由多个 **卷积层** 和 **残差块（residual blocks）** 组成，类似于 AlphaGo Zero，但结构进行了微调以适应更大棋盘（例如 19x19 围棋）。

#### 关键特性：

- 使用 **卷积层** 来处理局面（棋盘）数据。

- 用 **残差连接** 来避免深层网络的梯度消失问题。

- 引入 **平衡训练数据**，使得 AI 能够从各种局面中学习，而不仅仅是胜利的局面。


### **2. 蒙特卡洛树搜索（MCTS）**

KataGo 结合了蒙特卡洛树搜索（MCTS）来扩展其策略。MCTS 通过多次模拟局面的发展来评估最优的走法。具体来说，它通过以下步骤执行：

1. **选择阶段（Selection）**：在当前搜索树中，根据每个节点的统计信息选择最佳的子节点。

2. **扩展阶段（Expansion）**：如果选择的节点是未完全展开的节点，就会进行扩展，即添加新的子节点。

3. **模拟阶段（Simulation）**：进行随机模拟（自对弈），直到游戏结束。

4. **反向传播阶段（Backpropagation）**：根据模拟结果更新路径上的所有节点的统计信息，传播胜负信息。


#### KataGo 中的改进：

- 使用神经网络来引导 MCTS 搜索，而不是单纯依赖于随机模拟，显著提高了搜索效率和精度。

- 神经网络为 MCTS 提供了**策略网络**（Policy）和**价值网络**（Value）。策略网络用于指导搜索过程中需要探索哪些走法，而价值网络用于评估局面的好坏。


### **3. 强化学习与自我对弈**

KataGo 使用强化学习进行自我对弈训练。通过与自己对弈，KataGo 学习到了不断改进的策略。与 AlphaGo Zero 类似，KataGo 在训练过程中不依赖任何人工标签数据，而是通过自我对弈不断优化自己的策略。

### **4. 对比 AlphaGo Zero**

- **AlphaGo Zero** 采用的是**单一的价值网络**，用于评估局面并做出决策。而 KataGo 在此基础上加入了 **额外的评估**，如**多目标策略**，考虑棋盘的 **对称性** 和 **平衡性**，从而提升了策略的多样性和准确性。

- **KataGo** 的训练通过强化学习不断改进，具有比 AlphaGo 更加灵活的训练模式，适应不同棋盘和不同棋类。


---

### **5. KataGo 的关键模块**

1. **棋盘表示（Board Representation）**：使用一种高效的棋盘表示来减少内存占用和加速计算。

2. **神经网络推理（Network Inference）**：利用 GPU 高效推理，快速评估局面和生成策略。

3. **树搜索（Tree Search）**：基于 MCTS 进行多层次搜索，结合神经网络的建议进行优化。

4. **并行化**：KataGo 支持多线程和分布式计算，从而提高搜索效率和评估速度，尤其适合 **高性能计算平台**。


---

### **6. 实际应用**

KataGo 的应用非常广泛，包括：

- **围棋 AI**：作为围棋AI，KataGo 已经在许多比赛中表现出色，甚至击败了顶级围棋选手。

- **其他棋类**：尽管 KataGo 最初是为围棋设计的，但其框架也可以很容易地移植到其他棋类中，例如象棋、国际象棋等。


---

### **总结**

KataGo 的方法将深度神经网络与蒙特卡洛树搜索结合，并通过自我对弈和强化学习不断优化其策略。其通过引入神经网络来引导 MCTS、并行计算优化和高效的局面评估，达到了极高的棋类游戏水平。KataGo 不仅在围棋领域表现卓越，而且为未来的AI棋类开发提供了非常重要的技术参考。

---
```

- 除此之外我打算在此基础上辅助高性能计算，目前我在学习CUDA的相关内容，但是考虑到CUDA只能在nvidia显卡上运行，所以这部分我来搞定吧！

**Note:** 在神经网络的训练当中，最主要的就是神经网络的权重，我们需要通过训练、调整奖励函数，来不断更新权重，使得它逼近最优解，最终生成一个二进制文件（network_weights.bin），也需要提交到网站当中，相关的内容在书《深度学习入门》当中有所体现，可以看一下，内容不太多，在这个项目当中还是比较有用的，可以稍微看一下，了解个大概，有需要的时候知道要往哪方面想，再查资料或者问AI，虽然是用python写的，但是好在代码比较易读，我没有python基础的情况下辅助着AI可以弄懂。

---

以下作为记录开发日志，可以先把你的代码提交到仓库当中，然后稍微写写说明，方便我们维护版本

### 蒙特卡洛的优化

3.30 对蒙特卡洛算法进行了优化,主要是优化了评估函数,目前的权重设置依然存在问题,无法找到最优解.

> 目前存在的问题:
> 
> - 蒙特卡洛的搜索优化,现在的搜索深度最深是30,限制步数为100,后续要再进行优化,现在大部分时间只能搜到局部解,并且不是最优.
> - 不是最优解的情况可能有几个原因,但是我认为最重要的还是评估函数的调整,今天之内我会把评估函数调整好,使得它可以达到朴素算法的水平,没那么蠢
> - 现在的搜索深度依然不是极限,上午测试过之前的算法的极限大概就1500,当然随着评估函数的复杂化,这个极限会降低.

有一个newAtaxx.cpp,这个文件当中让Ai实现了我的一个想法,就是动态的棋盘格子权重评估,每次轮到我方时就会对当前局面进行计算,对某些"具有战略意义"的格子置成高权值,对某些无意义的格子置成低权值,我的判断具有战略意义的方法就是下在哪里会吃掉对方最多的棋子,这个方法肯定太草率,所以还要进行优化,不过这个可以当成一个优化思路

另外,我在查资料的时候发现了一个新的表示方法,我们现在的棋盘是用数组存储的棋盘,可以使用bitboard进行优化,采用位运算

3.31 对评估函数第一次尝试优化

> 使用了锦标赛加遗传进化算法的思想,建立一个种群,让种群内部进行锦标赛自博弈,并且第一次测试成功之后也是把测试的搜索算法换成了目前阶段使用的稳定的蒙特卡洛搜索
> 单线程锦标赛效率非常低,采用了OpenMP多线程并行计算,效率提升很高,但是时间耗费依旧很大.
> 优化完评估函数之后对接一下时间优化,第一版本的AtaxxAIbot就出炉了

我会把遗传进化的算法源代码上传到库当中,命名为newAtaxx,可以使用这个来优化评估函数.

## 第一代:

Generation 0 best fitness: 0
Best individual fitness: 0
Optimized parameters:
PieceScoreFactor: 18.4119
MobilityFactor: 4.33179
CornerFactor: 0
ConnectivityFactor: 0
ThreatFactor: 5.75072
StabilityFactor: 0.200662
CaptureFactor: 2.03004
PositionFactor: 2.49169

Generation 1 best fitness: 0
Best individual fitness: 0
Optimized parameters:
PieceScoreFactor: 14.5534
MobilityFactor: 0.578224
CornerFactor: 5.23451
ConnectivityFactor: 0
ThreatFactor: 7.45209
StabilityFactor: 1.11404
CaptureFactor: 6.24996
PositionFactor: 1.55627

## 发现了很多问题,准备记录一下

### 遗传算法的问题

    现在写的遗传算法会出现问题,具体就表现为无法筛选出最优子个体,一开始认为是胜率计算的问题或没有排序的问题,但是一直没有找到错误,后来发现第一版本的遗传算法会把父代个体全部移除,只剩下子代,并且父代都是相同的个体,现在修复了遗传算法,具体修复如下:
    > 1. 在每一代种群当中,会选出胜率前1/2继续加入下一代种群
    > 2. 在作为父代的标本当中,选择的是胜率前两名,并且在此基础上做变异

### 蒙特卡洛算法的问题

查找资料的时候,发现了一个问题,就是比较优秀的AIbot会简化评估函数,评估函数一般会非常简单,往往只有极小的参数,却达到很高效的效果,所以我想从参数部分入手,尽量简化计算.

好了现在我发现一个非常大的问题,就是蒙特卡洛算法有很大的问题,现在修改的评估函数,是作为蒙特卡洛算法当中的初始参数!! 对棋局的影响非常少


4.3晚间对蒙特卡洛搜索做了简单的修改,把蒙特卡洛当中的评估函数放到了正确的位置,放到服务器上运行,一方面作为速度尝试,另一方面也是想算出一代参数,查看是否可以使用.

4.4早10点, 在服务器上运行了10h,只运行了153把对局,没有达到预期,并且还开了多线程模拟,所以现在的算法非常耗时并且低效,原先的版本4h112把,慢了将近1/2,这也非常正常,因为把评估函数挪到了循环内部,每一次循环都会进行评估,慢了好多.

> 现在我有一个疑问,就是现在的蒙特卡洛模拟我设置了一个深度限制,这个深度是50,也就是理论往下计算50步,既然如此,为什么会出现那么蠢的走法,并且还不超时??很奇怪,我想我还是没有理解这个算法.

4.4中午, 已经照着AI的建议修复了一下蒙特卡洛模拟,高铁上网络不好,上传到github了,为了方便优化,现将修改之后的代码放到下面:

```cpp
// ----------------------------------一些优化---------------------------------
string hashState(const SimulationState& state){
    stringstream ss;
    for(int y = 0; y<7; y++){
        for(int x = 0; x < 7; x++){
            ss << state.grid[x][y] + 1;
        }
    }
    ss << ":" << state.currentPlayer;
    return ss.str();
}

// 优化之后的决策选择
pair< pair<int, int> ,pair<int, int> > selectBestMove(
    const vector<pair< pair<int, int>, pair<int, int> > >& moves,
    const SimulationState& state,
    map<string, NodeStats>& nodeStats,
    int alpha, int beta, int currBotColor){


    const double EXPLORATION_PARAM = 1.414;
    int bestMoveIdx = 0;
    double bestScore = state.currentPlayer == currBotColor ? -__DBL_MAX__ : __DBL_MAX__;
    for (size_t i = 0; i < moves.size(); i++){
        // 创建模拟状态
        SimulationState nextState = state;
        
        auto [fromPos, toPos] = moves[i];
        int x0 = fromPos.first, y0 = fromPos.second;
        int x1 = toPos.first, y1 = toPos.second;

        ProcStep(x0, y0, x1, y1, currBotColor);
        
        string HashState = hashState(nextState);
        auto stats = nodeStats[HashState];

        // UCB优化

        double exploitation = stats.visits == 0 ? 0 : stats.totalScore / stats.visits;
        double exploration = stats.visits == 0? 10000 :
            EXPLORATION_PARAM * sqrt(log(state.depth + 1 / stats.visits));

        double ucbScore;
        if (state.currentPlayer == currBotColor){
            //我方最大UCB
            ucbScore = exploitation + exploration;
            if(ucbScore > bestScore){
                bestScore = ucbScore;
                bestMoveIdx = i;
                alpha = max(alpha, (int)ucbScore);
                if(beta <= alpha) break;
            }

        }
        else{
            // 对方最小ucb
            ucbScore = exploitation - exploration;

            // minimax逻辑
            if(ucbScore < bestScore){
                bestScore = ucbScore;
                bestMoveIdx = i;
                beta = min(beta, (int)ucbScore);
                if (beta <= alpha) break;
            }
        }
    }

    return moves[bestMoveIdx];
}


// 优化后的蒙特卡洛模拟函数
int OptimizedMonteCarloSimulation(int startX, int startY, int resultX, int resultY, int simulations, int currBotColor) {
    int wins = 0;
    int originalGrid[7][7];
    int originalBlack = blackPieceCount, originalWhite = whitePieceCount;

    memcpy(originalGrid, gridInfo, sizeof(gridInfo));

    // 先执行当前走法
    ProcStep(startX, startY, resultX, resultY, currBotColor);

    totalPieces = blackPieceCount + whitePieceCount;
    int maxDepth = 30;
    if (totalPieces > 35) {
        maxDepth = 50;
        simulations = 500;
    } else if (totalPieces < 10) {
        maxDepth = 60;
    }

    for (int i = 0; i < simulations; i++) {
        SimulationState state;
        memcpy(state.grid, gridInfo, sizeof(state.grid));
        state.blackCount = blackPieceCount;
        state.whiteCount = whitePieceCount;
        state.currentPlayer = -currBotColor; // 对手回合
        state.depth = 0;

        map<string, NodeStats> nodeStats; // 使用字符串哈希表示局面

        while (state.depth < maxDepth) {
            // 生成所有合法走法
            vector<pair<pair<int, int>, pair<int, int> > > moves;
            int alpha = INT_MIN, beta = INT_MAX;

            for (int y0 = 0; y0 < 7; y0++) {
                for (int x0 = 0; x0 < 7; x0++) {
                    if (state.grid[x0][y0] == state.currentPlayer) {
                        for (int dir = 0; dir < 24; dir++) {
                            int x1 = x0 + delta[dir][0];
                            int y1 = y0 + delta[dir][1];
                            if (inMap(x1, y1) && state.grid[x1][y1] == 0) {
                                moves.push_back({{x0, y0}, {x1, y1}});
                            }
                        }
                    }
                }
            }

            if (moves.empty()) {
                state.currentPlayer = -state.currentPlayer;
                state.depth++;

                if (IsGameOver()) {
                    break;
                }
                continue;
            }

            // 选择最佳走法
            // auto bestMove = moves[0];
            // int bestScore = INT_MIN;
            // for (auto& move : moves) {
            //     int x0 = move.first.first, y0 = move.first.second;
            //     int x1 = move.second.first, y1 = move.second.second;

            //     // 模拟走法
            //     int tempGrid[7][7];
            //     memcpy(tempGrid, state.grid, sizeof(tempGrid));
            //     int tempBlack = state.blackCount, tempWhite = state.whiteCount;

            //     ProcStep(x0, y0, x1, y1, state.currentPlayer, true);

            //     // 评估当前局面
            //     int score = evaluateMoveType(x0, y0, x1, y1, state.currentPlayer);
            //     score += EvaluateSim(state.currentPlayer);

            //     if (score > bestScore) {
            //         bestScore = score;
            //         bestMove = move;
            //     }

            //     // 恢复局面
            //     memcpy(state.grid, tempGrid, sizeof(tempGrid));
            //     state.blackCount = tempBlack;
            //     state.whiteCount = tempWhite;
            // }

            // 更新的选择方法 
            auto bestMove = selectBestMove(moves, state, nodeStats, alpha, beta, currBotColor);
            auto [fromPos, toPos] = bestMove;

            // 执行走法
            int x0 = fromPos.first, y0 = fromPos.second;
            int x1 = toPos.first, y1 = toPos.second;
            ProcStep(x0, y0, x1, y1, state.currentPlayer);

            // 切换玩家
            state.currentPlayer = -state.currentPlayer;
            state.depth++;
        }

        // 评估最终局面
        int finalScore = (currBotColor == 1) ? 
            (state.blackCount - state.whiteCount) : 
            (state.whiteCount - state.blackCount);

        if (finalScore > 0)
            wins++;
    }

    // 恢复原始状态
    memcpy(gridInfo, originalGrid, sizeof(gridInfo));
    blackPieceCount = originalBlack;
    whitePieceCount = originalWhite;

    return wins;
}
```

 钟姐可以对以上算法再进行优化,我现在有疑问的点就在于:我们原来好像没有进行剪枝,现在才把剪枝加入,但是我写的剪枝算法只是比较简单的实现,还需要继续优化,还有外部的剪枝(调用蒙特卡洛之前),我觉得其实也可以剪枝一下,有些格子是一看就没有必要搜索的,可以提前判断这个格子的价值,选出最有价值的前几个格子进行模拟,又会优化时间.

> **note:** 但是这样如果剪不好我觉得会出现一种问题,就是有可能会把当前步数次优,但是总体最优的结果提前剪枝,陷入局部最优解,后续还需要再研究..


## 放弃蒙特卡洛了,使用了negamax算法,准备接入神经网络..