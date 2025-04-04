#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <random>
#include <omp.h>
#include <map> // For std::map
#include <sstream> // For std::stringstream

using namespace std;

// ----------------------------
// Ataxx 程序的全局变量及函数
// ----------------------------

//-----------------------------
// 用来做决策的全局变量

    // 记录每个节点的统计信息，用于 UCB 计算
    struct NodeStats {
        int visits = 0;
        double totalScore = 0.0;
    };

    // 用于跟踪每次模拟的状态
    struct SimulationState {
        int grid[7][7];
        int blackCount;
        int whiteCount;
        int currentPlayer;
        int depth;
    };
//-------------------------------
int winnerbot = 0;         // 胜利方
int totalPieces = 0;          // 棋盘上棋子总数
int blackPieceCount = 2, whitePieceCount = 2;
int gridInfo[7][7] = { 0 };   // 棋盘状态：先 x 后 y
static int delta[24][2] = {
    { 1,1 },{ 0,1 },{ -1,1 },{ -1,0 },
    { -1,-1 },{ 0,-1 },{ 1,-1 },{ 1,0 },
    { 2,0 },{ 2,1 },{ 2,2 },{ 1,2 },
    { 0,2 },{ -1,2 },{ -2,2 },{ -2,1 },
    { -2,0 },{ -2,-1 },{ -2,-2 },{ -1,-2 },
    { 0,-2 },{ 1,-2 },{ 2,-2 },{ 2,-1 }
};

// 初始位置权重矩阵（用于动态权重更新）
const int POSITION_WEIGHT[7][7] = {
    {10, -20, 10,  5, 10, -20, 10},
    {-20, -50, -2, -2, -2, -50, -20},
    {10,  -2,  5,  1,  5,  -2,  10},
    {5,   -2,  1,  0,  1,  -2,   5},
    {10,  -2,  5,  1,  5,  -2,  10},
    {-20, -50, -2, -2, -2, -50, -20},
    {10, -20, 10,  5, 10, -20, 10}
};

// 判断是否在地图内
inline bool inMap(int x, int y) {
    return x >= 0 && x < 7 && y >= 0 && y < 7;
}
// ----------------------------
// Move struct definition
struct Move {
    int x0, y0, x1, y1;
    Move(int startX, int startY, int endX, int endY) : x0(startX), y0(startY), x1(endX), y1(endY) {}
};

// ----------------------------
// 动态位置权重类（部分保持原有实现）
// 动态位置权重类（部分保持原有实现）
struct DynamicPositionWeight {
    int weights[7][7];
    
    void initialize() {
        for (int y = 0; y < 7; y++)
            for (int x = 0; x < 7; x++)
                weights[x][y] = POSITION_WEIGHT[x][y];
    }
    
    // 分阶段更新（保持原有逻辑）
    void updateEarlyGameWeights(int color) {
        for (int y = 2; y < 5; y++)
            for (int x = 2; x < 5; x++)
                weights[x][y] += 5;
        for (int y = 0; y < 7; y++) {
            for (int x = 0; x < 7; x++) {
                if (gridInfo[x][y] != 0) continue;
                bool isReachable = false;
                for (int y0 = 0; y0 < 7 && !isReachable; y0++) {
                    for (int x0 = 0; x0 < 7; x0++) {
                        if (gridInfo[x0][y0] != color) continue;
                        int dx = abs(x - x0), dy = abs(y - y0);
                        if (dx <= 2 && dy <= 2 && !(dx==0 && dy==0)) {
                            isReachable = true;
                            break;
                        }
                    }
                }
                if (!isReachable)
                    weights[x][y] -= 15;
            }
        }
    }
    
    void updateMidGameWeights(int color) {
        for (int y = 0; y < 7; y++) {
            for (int x = 0; x < 7; x++) {
                if (gridInfo[x][y] != 0) continue;
                int captureCount = 0;
                for (int dir = 0; dir < 8; dir++) {
                    int nx = x + delta[dir][0], ny = y + delta[dir][1];
                    if (inMap(nx, ny) && gridInfo[nx][ny] == -color)
                        captureCount++;
                }
                weights[x][y] += captureCount * captureCount * 3;
            }
        }
        for (int y0 = 0; y0 < 7; y0++) {
            for (int x0 = 0; x0 < 7; x0++) {
                if (gridInfo[x0][y0] != -color) continue;
                for (int dir = 0; dir < 8; dir++) {
                    int x = x0 + delta[dir][0], y = y0 + delta[dir][1];
                    if (inMap(x, y) && gridInfo[x][y] == 0)
                        weights[x][y] += 8;
                }
            }
        }
    }
    
    void updateLateGameWeights(int color) {
        for (int i = 0; i < 7; i++) {
            weights[0][i] += 10;
            weights[6][i] += 10;
            weights[i][0] += 10;
            weights[i][6] += 10;
        }
        weights[0][0] += 15; weights[0][6] += 15;
        weights[6][0] += 15; weights[6][6] += 15;
        for (int y = 0; y < 7; y++) {
            for (int x = 0; x < 7; x++) {
                if (gridInfo[x][y] != 0) continue;
                int captureCount = 0;
                for (int dir = 0; dir < 8; dir++) {
                    int nx = x + delta[dir][0], ny = y + delta[dir][1];
                    if (inMap(nx, ny) && gridInfo[nx][ny] == -color)
                        captureCount++;
                }
                weights[x][y] += captureCount * captureCount * 5;
            }
        }
    }
    
    void update(int totalPieces, int color);
    void updateCaptureAndMoveEfficiency(int color);
};

void DynamicPositionWeight::update(int totalPieces, int color) {
    initialize();
    if (totalPieces < 15) {
        updateEarlyGameWeights(color);
    } else if (totalPieces < 30) {
        updateMidGameWeights(color);
    } else {
        updateLateGameWeights(color);
    }
    updateCaptureAndMoveEfficiency(color);
}

void DynamicPositionWeight::updateCaptureAndMoveEfficiency(int color) {
    for (int y = 0; y < 7; y++) {
        for (int x = 0; x < 7; x++) {
            if (gridInfo[x][y] != 0) continue;
            int maxCaptureWithCopy = 0, maxCaptureWithJump = 0;
            bool canCopy = false, canJump = false;
            for (int y0 = 0; y0 < 7; y0++) {
                for (int x0 = 0; x0 < 7; x0++) {
                    if (gridInfo[x0][y0] != color) continue;
                    int dx = abs(x0 - x), dy = abs(y0 - y);
                    if (dx <= 1 && dy <= 1 && !(dx==0 && dy==0)) {
                        canCopy = true;
                        int captures = 0;
                        for (int dir = 0; dir < 8; dir++) {
                            int nx = x + delta[dir][0], ny = y + delta[dir][1];
                            if (inMap(nx, ny) && gridInfo[nx][ny] == -color)
                                captures++;
                        }
                        maxCaptureWithCopy = max(maxCaptureWithCopy, captures);
                    }
                    else if (dx <= 2 && dy <= 2 && !(dx==0 && dy==0)) {
                        canJump = true;
                        int captures = 0;
                        for (int dir = 0; dir < 8; dir++) {
                            int nx = x + delta[dir][0], ny = y + delta[dir][1];
                            if (inMap(nx, ny) && gridInfo[nx][ny] == -color)
                                captures++;
                        }
                        maxCaptureWithJump = max(maxCaptureWithJump, captures);
                    }
                }
            }
            if (canCopy && maxCaptureWithCopy > 0) {
                weights[x][y] += maxCaptureWithCopy * maxCaptureWithCopy * 4;
            }
            if (canJump && maxCaptureWithJump > 0) {
                int jumpBonus = maxCaptureWithJump * maxCaptureWithJump * 2;
                if (maxCaptureWithJump < 3)
                    jumpBonus -= 10;
                weights[x][y] += jumpBonus;
            }
            if (canCopy && canJump) {
                if (maxCaptureWithCopy >= maxCaptureWithJump - 1)
                    weights[x][y] += 15;
            }
            if ((x == 0 || x == 6 || y == 0 || y == 6))
                weights[x][y] += 12;
        }
    }
}

// 向指定方向移动坐标
inline bool MoveStep(int &x, int &y, int Direction) {
    x += delta[Direction][0];
    y += delta[Direction][1];
    return inMap(x, y);
}

// 落子操作，simulate 为 true 时用于模拟回滚
bool ProcStep(int x0, int y0, int x1, int y1, int color, bool simulate = false) {
    if (color == 0) return false;
    if (x1 == -1) return true;
    if (!inMap(x0, y0) || !inMap(x1, y1)) return false;
    if (gridInfo[x0][y0] != color) return false;
    
    int dx = abs(x0 - x1), dy = abs(y0 - y1);
    if ((dx == 0 && dy == 0) || dx > 2 || dy > 2) return false;
    if (gridInfo[x1][y1] != 0) return false;
    
    int originalGrid[7][7];
    int originalBlack = blackPieceCount, originalWhite = whitePieceCount;
    if (simulate)
        memcpy(originalGrid, gridInfo, sizeof(gridInfo));
    
    if (dx == 2 || dy == 2)
        gridInfo[x0][y0] = 0;
    else if (color == 1)
        blackPieceCount++;
    else
        whitePieceCount++;
    
    gridInfo[x1][y1] = color;
    int currCount = 0;
    for (int dir = 0; dir < 8; dir++) {
        int x = x1 + delta[dir][0], y = y1 + delta[dir][1];
        if (!inMap(x, y))
            continue;
        if (gridInfo[x][y] == -color) {
            currCount++;
            gridInfo[x][y] = color;
        }
    }
    if (currCount != 0) {
        if (color == 1) {
            blackPieceCount += currCount;
            whitePieceCount -= currCount;
        } else {
            whitePieceCount += currCount;
            blackPieceCount -= currCount;
        }
    }
    
    if (simulate) {
        memcpy(gridInfo, originalGrid, sizeof(gridInfo));
        blackPieceCount = originalBlack;
        whitePieceCount = originalWhite;
    }
    
    return true;
}

// 计算指定颜色合法走法数量
int numLegalMoves(int color) {
    int moveCount = 0;
    for (int y0 = 0; y0 < 7; y0++)
        for (int x0 = 0; x0 < 7; x0++) {
            if (gridInfo[x0][y0] != color) continue;
            for (int dir = 0; dir < 24; dir++) {
                int x1 = x0 + delta[dir][0], y1 = y0 + delta[dir][1];
                if (inMap(x1, y1) && gridInfo[x1][y1] == 0)
                    moveCount++;
            }
        }
    return moveCount;
}

// 计算稳定棋子数量
int countStablePieces(int color) {
    int stableCount = 0;
    const int corners[4][2] = { {0,0}, {0,6}, {6,0}, {6,6} };
    for (auto &corner : corners) {
        int x = corner[0], y = corner[1];
        if (gridInfo[x][y] == color)
            stableCount += 3;
    }
    for (int i = 1; i < 6; i++) {
        if (gridInfo[0][i] == color) stableCount++;
        if (gridInfo[6][i] == color) stableCount++;
        if (gridInfo[i][0] == color) stableCount++;
        if (gridInfo[i][6] == color) stableCount++;
    }
    for (int y = 1; y < 6; y++)
        for (int x = 1; x < 6; x++) {
            if (gridInfo[x][y] != color) continue;
            int sameColorNeighbors = 0;
            for (int dir = 0; dir < 8; dir++) {
                int nx = x + delta[dir][0], ny = y + delta[dir][1];
                if (inMap(nx, ny) && gridInfo[nx][ny] == color)
                    sameColorNeighbors++;
            }
            if (sameColorNeighbors >= 4)
                stableCount++;
        }
    return stableCount;
}

// ----------------------------
// 改进的增强连通性评估函数（考虑几何多样性）
int countConnectedPieces(int color) {
    bool visited[7][7] = {false};
    int totalConnectivityScore = 0;
    int directionDiversity[7][7] = {0};
    int geometricBonus[7][7] = {0};
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++) {
            if (gridInfo[x][y] != color)
                continue;
            bool hasDirection[8] = {false};
            int dirCount = 0;
            for (int dir = 0; dir < 8; dir++) {
                int nx = x + delta[dir][0], ny = y + delta[dir][1];
                if (inMap(nx, ny) && gridInfo[nx][ny] == color) {
                    hasDirection[dir] = true;
                    dirCount++;
                }
            }
            directionDiversity[x][y] = dirCount;
            for (int i = 0; i < 8; i++) {
                if (!hasDirection[i]) continue;
                for (int j = i+1; j < 8; j++) {
                    if (!hasDirection[j]) continue;
                    if (abs(i - j) != 4 && (i+j) % 8 != 0) {
                        geometricBonus[x][y] += 3;
                        break;
                    }
                }
                if (geometricBonus[x][y] > 0) break;
            }
        }
    for (int y = 0; y < 7; y++)
        for (int x = 0; x < 7; x++) {
            if (gridInfo[x][y] == color && !visited[x][y]) {
                int regionSize = 0, regionGeometricScore = 0, regionDirectionScore = 0;
                queue<pair<int, int>> q;
                q.push({x, y});
                visited[x][y] = true;
                regionSize++;
                regionGeometricScore += geometricBonus[x][y];
                regionDirectionScore += directionDiversity[x][y];
                while (!q.empty()) {
                    auto curr = q.front();
                    q.pop();
                    for (int dir = 0; dir < 8; dir++) {
                        int nx = curr.first + delta[dir][0], ny = curr.second + delta[dir][1];
                        if (inMap(nx, ny) && gridInfo[nx][ny] == color && !visited[nx][ny]) {
                            visited[nx][ny] = true;
                            q.push({nx, ny});
                            regionSize++;
                            regionGeometricScore += geometricBonus[nx][ny];
                            regionDirectionScore += directionDiversity[nx][ny];
                        }
                    }
                }
                int regionScore = regionSize * regionSize;
                regionScore += regionGeometricScore * 2;
                regionScore += (regionDirectionScore * 3) / regionSize;
                totalConnectivityScore += regionScore;
            }
        }
    return totalConnectivityScore;
}

// 计算对手威胁
int countOpponentThreats(int color) {
    int threatCount = 0;
    int opponentColor = -color;
    for (int y0 = 0; y0 < 7; y0++)
        for (int x0 = 0; x0 < 7; x0++) {
            if (gridInfo[x0][y0] != opponentColor)
                continue;
            for (int dir = 0; dir < 24; dir++) {
                int x1 = x0 + delta[dir][0], y1 = y0 + delta[dir][1];
                if (!inMap(x1, y1) || gridInfo[x1][y1] != 0)
                    continue;
                int captureCount = 0;
                for (int adjDir = 0; adjDir < 8; adjDir++) {
                    int ax = x1 + delta[adjDir][0], ay = y1 + delta[adjDir][1];
                    if (inMap(ax, ay) && gridInfo[ax][ay] == color)
                        captureCount++;
                }
                threatCount += captureCount * captureCount;
            }
        }
    return threatCount;
}

// ----------------------------
// 全局用于遗传优化的参数（待进化参数）
double g_pieceScoreFactor = 22.2316;
double g_mobilityFactor = 2.0002;
double g_connectivityFactor = 3.03559;
double g_threatFactor = 6.7919;
double g_captureFactor = 5.38024;

// 通过个体基因更新全局参数
void setGlobalParameters(const vector<double>& genes) {
    g_pieceScoreFactor = genes[0];
    g_mobilityFactor = genes[1];
    g_connectivityFactor = genes[2];
    g_threatFactor = genes[3];
    g_captureFactor = genes[4];
}

// 为自对奕设计的简化评估函数（使用全局参数）
double EvaluateSim(int color) {
    double score = 0;
    if (color == 1)
        score += g_pieceScoreFactor * (blackPieceCount - whitePieceCount);
    else
        score += g_pieceScoreFactor * (whitePieceCount - blackPieceCount);
    int myMoves = numLegalMoves(color);
    int oppMoves = numLegalMoves(-color);
    score += g_mobilityFactor * (myMoves - oppMoves);
    const int corners[4][2] = { {0,0}, {0,6}, {6,0}, {6,6} };
    int cornerScore = 0;
    // for (int i = 0; i < 4; i++) {
    //     int x = corners[i][0], y = corners[i][1];
    //     if (gridInfo[x][y] == color)
    //         cornerScore += 10;
    //     else if (gridInfo[x][y] == -color)
    //         cornerScore -= 10;
    // }
    //score += g_cornerFactor * cornerScore;
    score += g_connectivityFactor * countConnectedPieces(color);
    score -= g_threatFactor * countOpponentThreats(color);
    //score += g_stabilityFactor * countStablePieces(color);
    return score;
}

bool IsGameOver()
{
    if (blackPieceCount == 0 || whitePieceCount == 0) {
        // -1代表白色胜利,1代表黑色胜利
        winnerbot = (blackPieceCount == 0) ? -1 : 1;
        return true;
    }
    
    bool blackCanMove = false, whiteCanMove = false;
    for (int y0 = 0; y0 < 7; y0++)
    {
        for (int x0 = 0; x0 < 7; x0++)
        {
            if (gridInfo[x0][y0] == 1)
            {
                for (int dir = 0; dir < 24; dir++)
                {
                    int x1 = x0 + delta[dir][0];
                    int y1 = y0 + delta[dir][1];
                    if (inMap(x1, y1) && gridInfo[x1][y1] == 0)
                    {
                        blackCanMove = true;
                        break;
                    }
                }
            }
            else if (gridInfo[x0][y0] == -1)
            {
                for (int dir = 0; dir < 24; dir++)
                {
                    int x1 = x0 + delta[dir][0];
                    int y1 = y0 + delta[dir][1];
                    if (inMap(x1, y1) && gridInfo[x1][y1] == 0)
                    {
                        whiteCanMove = true;
                        break;
                    }
                }
            }
            if (blackCanMove && whiteCanMove)
                return false;
        }
    }
    if(!blackCanMove) {
        winnerbot = -1;
    } else if(!whiteCanMove) {
        winnerbot = 1;
    }
    else {
        winnerbot = 0;
    }
    return true;
}

int evaluateMoveType(int x0, int y0, int x1, int y1, int color)
{
    int moveScore = 0;
    int dx = abs(x0 - x1), dy = abs(y0 - y1);
    bool isCopy = (dx <= 1 && dy <= 1);
    
    totalPieces = blackPieceCount + whitePieceCount;
    
    if (isCopy)
    {
        moveScore += g_captureFactor * ((blackPieceCount-whitePieceCount) == 0 ? 1 : abs(blackPieceCount-whitePieceCount));
    }
    else
    {
        moveScore -= g_captureFactor * ((blackPieceCount-whitePieceCount) == 0 ? 1 : abs(blackPieceCount-whitePieceCount));
    }
    
    int formingTriangle = 0;
    bool hasDirection[8] = {false};
    for (int dir = 0; dir < 8; dir++)
    {
        int nx = x1 + delta[dir][0];
        int ny = y1 + delta[dir][1];
        if (inMap(nx, ny) && (gridInfo[nx][ny] == color || (nx == x0 && ny == y0)))
            hasDirection[dir] = true;
    }
    for (int i = 0; i < 8; i++) 
    {
        if (!hasDirection[i]) continue;
        for (int j = i + 1; j < 8; j++) 
        {
            if (!hasDirection[j] || abs(i - j) == 4) continue;
            formingTriangle++;
        }
    }
    moveScore += formingTriangle * 4;
    
    return moveScore;
}

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

// 简单的游戏模拟：双方轮流使用 decideMove() 选择走法，直至游戏结束
// 返回 1 表示黑胜，0 表示白胜，0.5 表示平局
// 修改后的 simulateGame 函数：使用蒙特卡洛模拟来选择走法
double simulateGame(const vector<double>& paramBlack, const vector<double>& paramWhite) {
    // 保存初始状态
    // 输出对奕双方的基因
    cout << "对奕双方的基因" << endl;
    cout << "Black Genes: ";
    for (double gene : paramBlack) {
        cout << gene << " ";
    }
    cout << endl;
    cout << "White Genes: ";
    for (double gene : paramWhite) {
        cout << gene << " ";
    }
    cout << endl;
    int initGrid[7][7];
    memcpy(initGrid, gridInfo, sizeof(gridInfo));
    int initBlack = blackPieceCount, initWhite = whitePieceCount;
    
    // 初始化棋盘：按照约定，初始角点布局
    memset(gridInfo, 0, sizeof(gridInfo));
    gridInfo[0][0] = gridInfo[6][6] = 1;
    gridInfo[6][0] = gridInfo[0][6] = -1;
    blackPieceCount = 2; whitePieceCount = 2;
    
    int currentColor = 1; // 黑先手
    int moveCount = 0;
    while (moveCount < 1000) { // 防止死循环
        // 根据当前回合设置全局参数
        if (currentColor == 1)
            setGlobalParameters(paramBlack);
        else
            setGlobalParameters(paramWhite);
        
        // 枚举当前颜色所有合法走法
        vector<Move> legalMoves;
        for (int y = 0; y < 7; y++) {
            for (int x = 0; x < 7; x++) {
                if (gridInfo[x][y] != currentColor) continue;
                for (int d = 0; d < 24; d++) {
                    int nx = x + delta[d][0];
                    int ny = y + delta[d][1];
                    if (inMap(nx, ny) && gridInfo[nx][ny] == 0) {
                        legalMoves.push_back({x, y, nx, ny});
                    }
                }
            }
        }
        if (legalMoves.empty()) { // 无合法走法则换手
            currentColor = -currentColor;
            moveCount++;
            continue;
        }
        
        // 对所有走法使用蒙特卡洛模拟进行评估，选择得分最高的走法
        int bestScore = -1000000;
        Move bestMove = legalMoves[0];
        int simulations = 100; // 默认模拟次数
        int totalPieces = blackPieceCount + whitePieceCount;
        if (totalPieces > 35)
            simulations = 150;
        // 对于当前走法，传入当前走子者颜色作为 currBotColor
        for (auto &m : legalMoves) {
            int score = OptimizedMonteCarloSimulation(m.x0, m.y0, m.x1, m.y1, simulations, currentColor);
            if (score > bestScore) {
                bestScore = score;
                bestMove = m;
            }
        }
        
        // 执行选定走法
        ProcStep(bestMove.x0, bestMove.y0, bestMove.x1, bestMove.y1, currentColor);
        currentColor = -currentColor;
        moveCount++;
        // 若双方均无合法走法，则结束游戏
        if (IsGameOver()) {
            break;
        }
    }
    
    // 计算游戏结果
    double result = 0.5; // 平局
    if (winnerbot == 1) {
        result = 1.0; // 黑胜
    } else if (winnerbot == -1) {
        result = 0.0; // 白胜
    }
    else if (winnerbot == 0) {
        result = 0.5; // 平局
    }
    // 输出游戏结果
    cout << result <<" ";
    cout << "Game Result: " << (result == 1.0 ? "Black Wins" : (result == 0.0 ? "White Wins" : "Draw")) << endl;
    cout << "Black Score: " << blackPieceCount << ", White Score: " << whitePieceCount << endl;
    cout << "胜者的基因:";
    if(result == 1.0) {
        for (double gene : paramBlack) {
            cout << gene << " ";
        }
    } else {
        for (double gene : paramWhite) {
            cout << gene << " ";
        }
    }
    cout << endl;

    // 恢复初始状态
    memcpy(gridInfo, initGrid, sizeof(gridInfo));
    blackPieceCount = initBlack;
    whitePieceCount = initWhite;
    return result;
}

// ----------------------------
// 遗传算法部分
struct Individual {
    vector<double> genes; // [pieceScoreFactor, mobilityFactor, cornerFactor, connectivityFactor, threatFactor]
    double fitness;
    Individual() : genes(8, 0.0), fitness(0.0) {}
};

std::mt19937 rng((unsigned)time(0));
double randomDouble(double minVal, double maxVal) {
    std::uniform_real_distribution<double> dist(minVal, maxVal);
    return dist(rng);
}

vector<double> defaultParams = {
    20.0, // g_pieceScoreFactor
    3.0,  // g_mobilityFactor
    3.0,  // g_connectivityFactor
    5.0,  // g_threatFactor
    5.0,  // g_captureFactor
};
vector<double> winner{g_pieceScoreFactor, g_mobilityFactor,  g_connectivityFactor, g_threatFactor, g_captureFactor};

random_device rd;
normal_distribution<double> dist(0.0, 2.0); // 均值0，标准差2的正态分布

vector<Individual> generateInitialPopulation(int popSize) {
    vector<Individual> population(popSize);
    for (int i = 0; i < popSize; i++) {
        if(i == popSize - 1) {
            population[i].genes = defaultParams; // 最后一个个体使用当前最优参数
            population[i].fitness = 0.0;
            break;
        } else {
            population[i].genes.resize(defaultParams.size());
        }
        for (size_t j = 0; j < defaultParams.size(); j++) {
            population[i].genes[j] = defaultParams[j] + dist(rng); // 在初始值上做扰动
        }
    }
    // 输出每个个体的基因型
    cout << "初始种群基因型:" << endl;
    for (const auto &ind : population) {
        cout << "个体基因: ";
        for (double gene : ind.genes) {
            cout << gene << " ";
        }
        cout << endl;
    }
    
    return population;
}

// 适应度评估：个体与其他若干对手进行自对奕，计算平均得分（1为胜，0为负，0.5为平局）
double evaluateIndividual(const Individual &ind, const vector<Individual>& opponents, int gamesPerOpponent = 2) {
    double totalScore = 0;
    int count = 0;

    #pragma omp parallel for reduction(+:totalScore, count)
    for (size_t i = 0; i < opponents.size(); i++) {
        const auto &opp = opponents[i];
        for (int g = 0; g < gamesPerOpponent; g++) {
            double result = simulateGame(ind.genes, opp.genes);
            totalScore += result;
            count++;
            result = 1.0 - simulateGame(opp.genes, ind.genes);
            totalScore += result;
            count++;
        }
    }
    cout << "原来的方式" << totalScore / count << endl;
    cout << "新的方式" << totalScore / double(count) << endl;
    return totalScore / double(count);
}

void evaluatePopulation(vector<Individual>& population) {
    // 对每个个体，让其与种群内其他个体对弈，计算平均胜率作为适应度
    #pragma omp parallel for
    for (size_t i = 0; i < population.size(); i++) {
        #pragma omp critical
        {
            cout << "Thread " << omp_get_thread_num() << " is processing individual " << i << endl;
        }
        vector<Individual> opponents;
        for (size_t j = 0; j < population.size(); j++) {
            if (i != j)
                opponents.push_back(population[j]);
        }
        population[i].fitness = evaluateIndividual(population[i], opponents, 1);
        cout << "gene:" << population[i].genes[0] << " " << population[i].genes[1] << " " << population[i].genes[2] << " "
             << population[i].genes[3] << " " << population[i].genes[4] << endl;
        cout << "fitness:" << population[i].fitness << " " << endl;
    }
}

// Individual tournamentSelection(const vector<Individual>& population, int tournamentSize) {
//     int popSize = population.size();
//     Individual best;
//     best.fitness = -1e9;
//     for (int i = 0; i < tournamentSize; i++) {
//         int idx = rng() % popSize;
//         if (population[idx].fitness > best.fitness)
//             best = population[idx];
        
//         cout << "best fittness:" << best.fitness << endl;
//         cout << "best gene:" << best.genes[0] << " " << best.genes[1] << " " << best.genes[2] << " "
//              << best.genes[3] << " " << best.genes[4] << " " << best.genes[5] << " "
//              << best.genes[6] << " " << best.genes[7] << endl;
//     }
//     return best;
// }

Individual crossover(const Individual& parent1, const Individual& parent2) {
    Individual child;
    int geneSize = parent1.genes.size();
    int crossPoint = rng() % geneSize;
    for (int i = 0; i < geneSize; i++) {
        if (i < crossPoint)
            child.genes[i] = parent1.genes[i];
        else
            child.genes[i] = parent2.genes[i];
    }
    return child;
}

void mutate(Individual& ind, double mutationRate, double mutationStdDev) {
    std::normal_distribution<double> dist(0.0, mutationStdDev);
    for (double &gene : ind.genes) {
        if (randomDouble(0.0, 1.0) < mutationRate) {
            gene += dist(rng);
            gene = max(0.0, min(100.0, gene));
        }
    }
}

// ----------------------------
// 主函数：利用自对奕和遗传算法更新评估函数参数
int main() {
    // 动态设置线程数为最大线程数
    int maxThreads = omp_get_max_threads();
    omp_set_num_threads(maxThreads);
    cout << "Using " << maxThreads << " threads." << endl;

    const int populationSize = 3;
    const int generations = 3;
    const int tournamentSize = 3;
    const double mutationRate = 0.3;
    const double mutationStdDev = 5.0;
    const int offspringCount = populationSize;
    cout << generations<<endl;
    
    vector<Individual> population = generateInitialPopulation(populationSize);
    
    for (int gen = 0; gen < generations; gen++) {
        cout << gen << endl << endl;
        evaluatePopulation(population);

        // 按照fitness从大到小排序
        sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });

        vector<Individual> newPopulation;

        // 保留前generations/2个元素
        int eliteCount = generations / 2;
        for (int i = 0; i < eliteCount; i++) {
            newPopulation.push_back(population[i]);
        }

        // 父母选择为fitness排序的前两名
        Individual parent1 = population[0];
        Individual parent2 = population[1];
        cout << "parent1 fitness: " << parent1.fitness << endl;
        cout << "parent1 gene: " << parent1.genes[0] << " " << parent1.genes[1] << " " << parent1.genes[2] << " "
             << parent1.genes[3] << " " << parent1.genes[4] << endl;
        cout << "parent2 fitness: " << parent2.fitness << endl;
        cout << "parent2 gene: " << parent2.genes[0] << " " << parent2.genes[1] << " " << parent2.genes[2] << " "
             << parent2.genes[3] << " " << parent2.genes[4] << endl;

        // 用child补全剩余的个体
        while (newPopulation.size() < offspringCount) {
            Individual child = crossover(parent1, parent2);
            mutate(child, mutationRate, mutationStdDev);
            newPopulation.push_back(child);
        }

        // 输出新个体的基因型
        cout << "新个体基因: ";
        int idx = 1;
        for (auto& ind : newPopulation) {
            cout << "个体" << idx++ << ": ";
            cout << "fitness: " << ind.fitness << " ";
            cout << "基因: ";
            cout << ind.genes[0] << " " << ind.genes[1] << " " << ind.genes[2] << " "
                 << ind.genes[3] << " " << ind.genes[4] << " " << endl;
        }
        population = newPopulation;

        auto bestIt = max_element(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness < b.fitness;
        });
        cout << endl << "Generation " << gen << " best fitness: " << bestIt->fitness << endl;
        cout << "Best individual fitness: " << bestIt->fitness << endl;
        cout << "Optimized parameters:" << endl;
        cout << "PieceScoreFactor: " << bestIt->genes[0] << endl;
        cout << "MobilityFactor: " << bestIt->genes[1] << endl;
        cout << "CornerFactor: " << bestIt->genes[2] << endl;
        cout << "ConnectivityFactor: " << bestIt->genes[3] << endl;
        cout << "ThreatFactor: " << bestIt->genes[4] << endl;
    }
    
    evaluatePopulation(population);
    auto bestIt = max_element(population.begin(), population.end(), [](const Individual& a, const Individual& b){
        return a.fitness < b.fitness;
    });
    cout << endl << "结束!!" << "Best individual fitness: " << bestIt->fitness << endl;
    cout << "Optimized parameters:" << endl;
    cout << "PieceScoreFactor: " << bestIt->genes[0] << endl;
    cout << "MobilityFactor: " << bestIt->genes[1] << endl;
    cout << "CornerFactor: " << bestIt->genes[2] << endl;
    cout << "ConnectivityFactor: " << bestIt->genes[3] << endl;
    cout << "ThreatFactor: " << bestIt->genes[4] << endl;
    
    return 0;
}

