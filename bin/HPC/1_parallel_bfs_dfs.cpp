/*
 * Parallel BFS & DFS using OpenMP
 * Compile: g++ -fopenmp -o parallel_bfs_dfs 1_parallel_bfs_dfs.cpp
 * Run    : ./parallel_bfs_dfs
 */
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }

    // Parallel BFS
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        
        visited[start] = true;
        q.push(start);
        
        cout << "BFS: ";
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            cout << v << " ";

            #pragma omp parallel for
            for (int i = 0; i < adj[v].size(); i++) {
                int neighbor = adj[v][i];
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
        cout << endl;
    }

    // Parallel DFS
    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        cout << "DFS: ";
        
        #pragma omp parallel
        #pragma omp single
        parallelDFSUtil(start, visited);
        
        cout << endl;
    }

private:
    void parallelDFSUtil(int v, vector<bool>& visited) {
        bool should_process = false;
        #pragma omp critical
        {
            if (!visited[v]) {
                visited[v] = true;
                cout << v << " ";
                should_process = true;
            }
        }

        if (should_process) {
            #pragma omp parallel for
            for (int i = 0; i < adj[v].size(); i++) {
                parallelDFSUtil(adj[v][i], visited);
            }
        }
    }
};

int main() {
    Graph g(7);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);

    cout << "Threads available: " << omp_get_max_threads() << "\n\n";
    
    g.parallelDFS(0);
    g.parallelBFS(0);

    return 0;
}