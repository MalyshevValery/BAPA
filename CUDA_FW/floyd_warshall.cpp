//
// Created by malyshev on 06/09/2019.
//

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstring>
#include <thread>

#define PLACEHOLDER 100000
using namespace std;

void seq_fu(int n, int *matrix) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i * n + j] = min(matrix[i * n + j], matrix[i * n + k] + matrix[k * n + j]);
            }
        }
    }
}

void block_2D_fu(int n, int r, int *matrix) {
    if (n % r != 0) {
        cout << "Not effective block" << endl;
        return;
    }
    int nn = n / r;
    for (int k = 0; k < n; k++) {
        //Thread creation
        thread *threads = new thread[nn * nn];
        for (int n_block = 0; n_block < nn * nn; n_block++) {
            threads[n_block] = thread([=](int id) -> void {
                int ni = id / nn;
                int nj = id % nn;
                for (int i = ni * r; i < (ni + 1) * r; i++) {
                    for (int j = nj * r; j < (nj + 1) * r; j++) {
                        matrix[i * n + j] = min(matrix[i * n + j], matrix[i * n + k] + matrix[k * n + j]);
                    }
                }
                return;
            }, n_block);
        }
        //join and wait
        for (int i = 0; i < nn * nn; i++)
            threads[i].join();
    }
}

void block_3D_fu(int n, int r, int *matrix) {
    if (n % r != 0) {
        cout << "Not effective block" << endl;
        return;
    }
    const int nn = n / r;
    for (int nk = 0; nk < nn; nk++) {
        //I-block
        for (int k = nk * r; k < (nk + 1) * r; k++) {
            for (int i = nk * r; i < (nk + 1) * r; i++) {
                for (int j = nk * r; j < (nk + 1) * r; j++) {
                    matrix[i * n + j] = min(matrix[i * n + j], matrix[i * n + k] + matrix[k * n + j]);
                }
            }
        }

        //SD-blocks threads
        thread *threads = new thread[nn * 2 - 2];
        int t_counter = 0;
        //SD-block line
        for (int nj = 0; nj < nn; nj++) {
            if (nj == nk)
                continue;
            threads[t_counter++] = thread([=](int nj) -> void {
            for (int k = nk * r; k < (nk + 1) * r; k++) {
                for (int i = nk * r; i < (nk + 1) * r; i++) {
                    for (int j = nj * r; j < (nj + 1) * r; j++) {
                        matrix[i * n + j] = min(matrix[i * n + j], matrix[i * n + k] + matrix[k * n + j]);
                    }
                }
            }
            }, nj);
        }
        //SD-blocks column
        for (int ni = 0; ni < nn; ni++) {
            if (ni == nk)
                continue;
            threads[t_counter++] = thread([=](int ni) -> void {
                for (int k = nk * r; k < (nk + 1) * r; k++) {
                    for (int i = ni * r; i < (ni + 1) * r; i++) {
                        for (int j = nk * r; j < (nk + 1) * r; j++) {
                            matrix[i * n + j] = min(matrix[i * n + j], matrix[i * n + k] + matrix[k * n + j]);
                        }
                    }
                }
            }, ni);
        }

        for (int i = 0; i < t_counter; i++)
            threads[i].join();

        //DD-blocks
        threads = new thread[(nn - 1) * (nn - 1)];
        t_counter = 0;

        for (int ni = 0; ni < nn; ni++) {
            for (int nj = 0; nj < nn; nj++) {
                if (nj == nk || ni == nk)
                    continue;
                threads[t_counter++] = thread([=](int ni, int nj) -> void {
                    for (int i = ni * r; i < (ni + 1) * r; i++) {
                        for (int j = nj * r; j < (nj + 1) * r; j++) {
                            for (int k = nk * r; k < (nk + 1) * r; k++) {
                                matrix[i * n + j] = min(matrix[i * n + j], matrix[i * n + k] + matrix[k * n + j]);
                            }
                        }
                    }
                }, ni, nj);
            }
        }
        for (int i = 0; i < t_counter; i++)
            threads[i].join();
    }
}

/*
 * 1. filename
 */
int main(int argc, char **argv) {
    if (argc < 2)
        cout << "No input file";
    else if (argc < 3)
        cout << "No output file";
    else if (argc < 4)
        cout << "No type";
    struct timespec start, finish;
    double elapsed;
    cout.precision(3);
    string type = argv[3];
    int r = 1;
    if (argc == 5) {
        r = stoi(string(argv[4]));
    }

    ifstream fin(argv[1]);
    int n;
    fin >> n;
    cout << "Reading... ";
    clock_gettime(CLOCK_MONOTONIC, &start);
    int *matrix = new int[n * n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fin >> matrix[i * n + j];
            if (matrix[i * n + j] == -1)
                matrix[i * n + j] = PLACEHOLDER;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << fixed << elapsed << "s" << endl;

    clock_gettime(CLOCK_MONOTONIC, &start);
    if (strcmp(type.c_str(), "seq") == 0) {
        cout << "Sequential FU... ";
        seq_fu(n, matrix);
    } else if (strcmp(type.c_str(), "block_2d") == 0) {
        cout << "2D block FU... ";
        block_2D_fu(n, r, matrix);
    } else if (strcmp(type.c_str(), "block_3d") == 0) {
        cout << "3D block FU... ";
        block_3D_fu(n, r, matrix);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << fixed << elapsed << "s" << endl;

    ofstream fout(argv[2]);
    cout << "Writing... ";
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << matrix[i * n + j] << "\t";
        }
        fout << endl;
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << fixed << elapsed << "s" << endl;
    return 0;
}