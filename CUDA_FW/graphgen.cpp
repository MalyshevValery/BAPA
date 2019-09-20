#include<iostream>
#include <fstream>
#include <time.h>
#include<stdlib.h>

using namespace std;

#define MAX_WEIGHT 100
#define OUT_THR 10


// A function to generate random graph.
int **GenerateRandGraphs(int NOE, int NOV) {
    int **matrix = new int *[NOV];
    for (int i = 0; i < NOV; i++) {
        matrix[i] = new int[NOV];
    }

    int i, j;
    i = 0;
    // Build a connection between two random vertex.
    cout.precision(2);
    int counter = 0;
    while (i < NOE) {
        int x = rand() % NOV;
        int y = rand() % NOV;
        int w = rand() % MAX_WEIGHT + 1;

        if (x == y || matrix[x][y] != 0)
            continue;
        matrix[x][y] = w;
        i++;
        counter++;

        if (counter >= NOE / OUT_THR){
            counter = 0;
            cout << fixed << 100.0 * i / NOE << "%" << endl;
        }
    }
    cout << "100.00%" << endl;

    cout << "Fixing zeros..." << endl;
    for (int i = 0; i < NOV; i++) {
        for (int j = 0; j < NOV; j++){
            if (i != j && matrix[i][j] == 0) {
                matrix[i][j] = -1;
            }
        }
    }

    return matrix;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Wrong number of arguments";
        return 1;
    }
    srand(time(NULL));
    const string v_str = string(argv[1]);
    const string load_str = string(argv[2]);
    const string filename = string(argv[3]);

    int v = stoi(v_str);
    float load = stof(load_str);

    cout << "Random graph generation: " << endl;
    cout << "The graph has " << v << " vertexes." << endl;
    int e = load * v * (v - 1);
    cout << "\nThe graph has " << e << " edges." << endl;

    // A function to generate a random undirected graph with e edges and v vertexes.
    int **graph = GenerateRandGraphs(e, v);
    cout << "Generated" << endl;
    //Output
    cout << "Saving matrix..." << endl;
    ofstream fout(filename);
    fout << v << endl;
    for (int i = 0; i < v; i++) {
        for (int j = 0; j < v; j++)
            fout << graph[i][j] << "\t";
        fout << std::endl;
    }
    fout.close();
    cout << "Writing graph to file is finished" << std::endl;
    return 0;
}