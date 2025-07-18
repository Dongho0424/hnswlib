#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "../../hnswlib/hnswlib.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};



/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

// =================================================================
// SIFT-1M (float)에 맞게 수정된 헬퍼 함수들
// =================================================================

// GT 파싱 함수 (float* massQ 타입 사용)
static void
get_gt_f(
    unsigned int *massQA,
    float *massQ,
    size_t qsize,
    size_t k_gnd, // GT에 저장된 이웃의 수 (100)
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) { // recall@k에서 k
    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[k_gnd * i + j]);
        }
    }
}

// Recall 계산 함수 (float* massQ, HierarchicalNSW<float> 사용)
static float
test_approx_f(
    float *massQ,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    size_t correct = 0;
    size_t total = 0;

    for (int i = 0; i < qsize; i++) {
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

// ef에 따른 recall 테스트 함수 (float* massQ, HierarchicalNSW<float> 사용)
static void
test_vs_recall_f(
    float *massQ,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    // vector<size_t> efs;
    // for (int i = k; i < 30; i++) efs.push_back(i);
    // for (int i = 30; i < 100; i += 10) efs.push_back(i);
    // for (int i = 100; i < 500; i += 40) efs.push_back(i);
    vector<size_t> efs = {10, 30, 50, 70, 100}; 

    // for (size_t ef : efs) {
    //     appr_alg.setEf(ef);
    //     StopW stopw = StopW();

    //     float recall = test_approx_f(massQ, qsize, appr_alg, vecdim, answers, k);
    //     float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

    //     cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
    //     if (recall > 1.0) {
    //         cout << recall << "\t" << time_us_per_query << " us\n";
    //         break;
    //     }
    // }
    cout << "Ls" << "\t" << "QPS" << "\t\t" << "Recall@10" << "\t" << "Mean Latency (us)" << endl;
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx_f(massQ, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        double displayed_qps = 1000000 * qsize / stopw.getElapsedTimeMicro();

        cout << ef << "\t" << displayed_qps << "\t\t" << recall * 100 << "\t\t" << time_us_per_query << " us" << endl;
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

// =================================================================
// sift_test1M 함수 (메인 로직)
// =================================================================
void sift_test1m() {
    // CHANGED: SIFT-1M 데이터셋 파라미터
    int M = 32;
    int efConstruction = 100;
    size_t vecsize = 1000000; // 1M base vectors
    size_t qsize = 10000;     // 10K query vectors
    size_t vecdim = 128;      // SIFT vector dimension
    size_t k_gnd = 100;       // Ground truth에는 100개의 이웃이 있음

    // CHANGED: 파일 경로
    char path_index[1024];
    const char *path_data = "../sift1m/sift/sift_base.fvecs";
    const char *path_q = "../sift1m/sift/sift_query.fvecs";
    const char *path_gt = "../sift1m/sift/sift_groundtruth.ivecs";
    snprintf(path_index, sizeof(path_index), "sift1m_M%d_ef%d.bin", M, efConstruction);

    // CHANGED: 데이터 타입을 float으로 변경
    float *massb = new float[vecdim];

    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * k_gnd];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        if (t != k_gnd) {
            cout << "File " << path_gt << " contains " << t << " nearest neighbors per query, but expected " << k_gnd << endl;
            return;
        }
        inputGT.read((char *) (massQA + k_gnd * i), t * sizeof(int));
    }
    inputGT.close();

    cout << "Loading queries:\n";
    float *massQ = new float[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);
    for (int i = 0; i < qsize; i++) {
        int in;
        inputQ.read((char *) &in, 4);
        if (in != vecdim) {
            cout << "File error: query dimension " << in << " does not match expected " << vecdim << endl;
            return;
        }
        inputQ.read((char *) (massQ + i * vecdim), in * sizeof(float));
    }
    inputQ.close();

    // CHANGED: L2Space (float용) 및 HierarchicalNSW<float> 사용
    L2Space l2space(vecdim);
    HierarchicalNSW<float> *appr_alg;

    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, path_index, false);
    } else {
        cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction);
        ifstream input(path_data, ios::binary);
        StopW stopw_full = StopW();

        for (int i = 0; i < vecsize; i++) {
            int in;
            input.read((char *) &in, 4);
            if (in != vecdim) {
                cout << "File error: base vector dimension " << in << " does not match expected " << vecdim << " at vector " << i << endl;
                return;
            }
            input.read((char *)massb, in * sizeof(float));
            appr_alg->addPoint((void *)massb, (size_t)i);

            if (i % 100000 == 0) {
                cout << i / 10000.0 << " %\n";
            }
        }
        input.close();
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << " seconds\n";
        appr_alg->saveIndex(path_index);
    }

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = 10; // recall@10을 테스트
    cout << "Parsing gt:\n";
    get_gt_f(massQA, massQ, qsize, k_gnd, answers, k);

    cout << "Loaded gt\n";
    test_vs_recall_f(massQ, qsize, *appr_alg, vecdim, answers, k);

    delete[] massQA;
    delete[] massQ;
    delete[] massb;
    delete appr_alg;

    return;
}