#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <cblas.h>
#include <windows.h>

using namespace std;
using namespace chrono;

using Comp = complex<float>;

struct Matr {
    int n;
    vector<Comp> d;

    Matr(int s) : n(s), d(s*s) {}

    Comp get(int i, int j) const { return d[i * n + j]; }
    void set(int i, int j, Comp x) { d[i * n + j] = x; }
};

void randomFill(Matr& M) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-10, 10);
    for (int i = 0; i < M.n * M.n; i++) {
        M.d[i] = Comp(dis(gen), dis(gen));
    }
}

Matr mul1(const Matr& A, const Matr& B) {
    int n = A.n;
    Matr C(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Comp sum = 0;
            for (int k = 0; k < n; k++) {
                sum = sum + A.get(i, k) * B.get(k, j);
            }
            C.set(i, j, sum);
        }
    }
    return C;
}

Matr mul2(const Matr& A, const Matr& B) {
    int n = A.n;
    Matr C(n);

    float alpha[2] = {1, 0};
    float beta[2] = {0, 0};

    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, alpha,
                (float*)A.d.data(), n,
                (float*)B.d.data(), n,
                beta, (float*)C.d.data(), n);
    return C;
}

Matr mul3(const Matr& A, const Matr& B) {
    int n = A.n;
    Matr C(n);

    const Comp* a = A.d.data();
    const Comp* b = B.d.data();
    Comp* c = C.d.data();

    #pragma omp parallel for
    for (int i = 0; i < n * n; i++) {
        c[i] = 0;
    }

    int block = 128;

    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < n; i += block) {
        for (int j = 0; j < n; j += block) {
            for (int k = 0; k < n; k += block) {
                int ie = min(i + block, n);
                int je = min(j + block, n);
                int ke = min(k + block, n);

                for (int ii = i; ii < ie; ii++) {
                    for (int kk = k; kk < ke; kk++) {
                        Comp aik = a[ii * n + kk];
                        #pragma omp simd
                        for (int jj = j; jj < je; jj++) {
                            c[ii * n + jj] = c[ii * n + jj] + aik * b[kk * n + jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}

double test(const string& name, Matr& A, Matr& B, Matr (*f)(const Matr&, const Matr&)) {
    cout << "\n=== " << name << " ===" << endl;

    auto start = high_resolution_clock::now();
    Matr C = f(A, B);
    auto end = high_resolution_clock::now();

    double t = duration<double>(end - start).count();
    double mflops = 2LL * A.n * A.n * A.n / t / 1000000.0;

    cout << "Время: " << t << " сек" << endl;
    cout << "MFLOPS: " << mflops << endl;

    return mflops;
}

int main() {
    SetConsoleCP(65001);
    SetConsoleOutputCP(65001);
    const int N = 2048;

    cout << "Никитин Савелий Сергеевич, группа 020303-АИСа-о25" << endl;

    Matr A(N), B(N);

    randomFill(A);
    randomFill(B);
    mul3(A, B);

    double a = test("ВАРИАНТ 1: ", A, B, mul1);
    double b = test("ВАРИАНТ 2: ", A, B, mul2);
    double c = test("ВАРИАНТ 3: ", A, B, mul3);

    cout << "Вариант 1: " << (a / b * 100) << "% от BLAS" << endl;
    cout << "Вариант 2: 100% (эталон)" << endl;
    cout << "Вариант 3: " << (c / b * 100) << "% от BLAS" << endl;


    return 0;
}
