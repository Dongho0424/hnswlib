#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "../../hnswlib/hnswlib.h"


#include <unordered_set>

using namespace std;
using namespace hnswlib;

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


// static void
// get_gt(
//     unsigned int *massQA,
//     unsigned char *massQ,
//     unsigned char *mass,
//     size_t vecsize,
//     size_t qsize,
//     L2SpaceI &l2space,
//     size_t vecdim,
//     vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
//     size_t k) {
//     (vector<std::priority_queue<std::pair<int, labeltype >>>(qsize)).swap(answers);
//     DISTFUNC<int> fstdistfunc_ = l2space.get_dist_func();
//     cout << qsize << "\n";
//     for (int i = 0; i < qsize; i++) {
//         for (int j = 0; j < k; j++) {
//             answers[i].emplace(0.0f, massQA[1000 * i + j]);
//         }
//     }
// }

static float
test_approx(
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) 
{
    size_t correct = 0;
    size_t total = 0;
    // uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        std::priority_queue<std::pair<int, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<int, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            } else {
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void
test_vs_recall(
    unsigned char *massQ,
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<int> &appr_alg,
    size_t vecdim,
    vector<std::priority_queue<std::pair<int, labeltype>>> &answers,
    size_t k) {
    // search list size L
    // vector<size_t> efs; 
    vector<size_t> efs = {10, 30, 50, 70, 100}; 
    // for (int i = k; i < 30; i++) {
    //     efs.push_back(i);
    // }
    // for (int i = 30; i < 100; i += 10) {
    //     efs.push_back(i);
    // }
    // for (int i = 100; i < 500; i += 40) {
    //     efs.push_back(i);
    // }
    //   Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
    cout << "Ls" << "\t" << "QPS" << "\t\t" << "Recall@10" << "\t" << "Mean Latency (us)" << endl;
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
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

void bigann_10m_dongho() {
    // 데이터셋 크기를 10M으로 설정
    int subset_size_milllions = 10;
    int efConstruction = 100; // search list size when indexing graph
    // 상위 계층(layer > 0)에서의 최대 이웃 수, M0 = 2 * M 
    // DiskANN의 R (diameter bound와 같다)
    int M = 32; 
    // Ground Truth의 이웃 개수 (bigann-10M은 100개)
    // DiskANN에서는 gt_dim이다.
    size_t gt_k;
    size_t k = 10; // k-recall@k

    size_t vecsize = subset_size_milllions * 1000000;

    size_t qsize = 10000;
    size_t vecdim = 128;
    char path_index[1024];
    char path_gt[1024];

    const char *path_q = "../bigann10m/query.public.10K.u8bin";
    const char *path_data = "../bigann10m/base.1B.u8bin.crop_nb_10000000";

    snprintf(path_index, sizeof(path_index), "bigann10m_ef_%d_M_%d.bin", efConstruction, M);
    snprintf(path_gt, sizeof(path_gt), "../bigann10m/bigann-%dM", subset_size_milllions);

    unsigned char *massb = new unsigned char[vecdim];

    /* gt file loading */

    cout << "Loading GT in DiskANN format:\n";
    ifstream inputGT(path_gt, ios::binary);
    if (!inputGT) {
        cout << "Error: Cannot open GT file at " << path_gt << endl;
        return;
    }

    // DiskANN 형식에 따라 파일 헤더(전체 쿼리 수, 이웃 개수)를 먼저 읽습니다.
    int file_qsize, file_gt_k;
    inputGT.read((char *)&file_qsize, sizeof(int));
    inputGT.read((char *)&file_gt_k, sizeof(int));

    cout << "GT file metadata: #queries = " << file_qsize << ", #GT_neighbors = " << file_gt_k << endl;

    // 파일에서 읽은 쿼리 수가 코드의 qsize와 일치하는지 확인합니다.
    if (file_qsize != qsize) {
        cout << "Error: Mismatch in number of queries. Code expects " << qsize
             << ", but GT file has " << file_qsize << "." << endl;
        return;
    }

    // 파일에서 직접 읽은 이웃 개수(k)를 사용합니다.
    gt_k = file_gt_k; 

    // 파일 메타데이터에 맞춰 메모리를 할당합니다.
    unsigned int *massQA = new unsigned int[qsize * gt_k];

    // 모든 Ground Truth ID 데이터를 한 번에 읽어옵니다.
    cout << "Reading " << (long long)qsize * gt_k << " GT IDs..." << endl;
    inputGT.read((char *)massQA, (long long)qsize * gt_k * sizeof(unsigned int));

    if (!inputGT) {
        cout << "Error reading GT data from file. The file might be smaller than expected." << endl;
        delete[] massQA;
        return;
    }
    inputGT.close();

    /* query loading */

    cout << "Loading queries:\n";
    unsigned char *massQ = new unsigned char[qsize * vecdim];
    ifstream inputQ(path_q, ios::binary);

    // --- NEW: 파일 맨 앞에서 헤더를 한 번만 읽습니다. ---
    int file_vecdim;
    inputQ.read((char *)&file_qsize, sizeof(int));
    inputQ.read((char *)&file_vecdim, sizeof(int));

    // 헤더 정보가 예상과 맞는지 확인
    if (file_qsize != qsize || file_vecdim != vecdim) {
        cout << "Query file header mismatch!" << endl;
        cout << "Expected qsize=" << qsize << ", file has " << file_qsize << endl;
        cout << "Expected vecdim=" << vecdim << ", file has " << file_vecdim << endl;
        return;
    }

    for (int i = 0; i < qsize; i++) {
        inputQ.read((char *) massb, vecdim);
        for (int j = 0; j < vecdim; j++) {
            massQ[i * vecdim + j] = massb[j];
        }
    }
    inputQ.close();

    unsigned char *mass = new unsigned char[vecdim];
    ifstream input(path_data, ios::binary);
    int in = 0;
    L2SpaceI l2space(vecdim);

    HierarchicalNSW<int> *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<int>(&l2space, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } 
    else {
        cout << "Building index..." << endl;
        
        // --- 1. Base 데이터를 메모리로 전부 로딩 (싱글 스레드) ---
        cout << "Pre-loading base data into memory..." << endl;
        ifstream input(path_data, ios::binary);
        if (!input) {
            cout << "Could not open base data file: " << path_data << endl;
            return;
        }

        // 헤더 읽기
        int file_vecsize, file_vecdim;
        input.read((char *)&file_vecsize, sizeof(int));
        input.read((char *)&file_vecdim, sizeof(int));

        if (file_vecsize != vecsize || file_vecdim != vecdim) {
            cout << "Base data file header mismatch!" << endl;
            return;
        }

        // 전체 base 데이터를 저장할 메모리 공간 할당
        unsigned char* all_base_data = new unsigned char[(long long)vecsize * vecdim];
        input.read((char *)all_base_data, (long long)vecsize * vecdim);
        input.close();
        cout << "Base data loaded." << endl;

        // --- 2. 메모리에 있는 데이터를 사용해 병렬로 인덱스 생성 ---
        appr_alg = new HierarchicalNSW<int>(&l2space, vecsize, M, efConstruction);
        StopW stopw_full = StopW();
        
        size_t report_every = 100000;
        atomic<size_t> progress_counter(0); // atomic으로 스레드 안전한 카운터 사용
        StopW progress_timer = StopW();

        #pragma omp parallel for
        for (int i = 0; i < vecsize; i++) {
            // 이제 각 스레드는 파일이 아닌 메모리에서 자신의 데이터를 가져옴
            // i번 스레드는 정확히 i번 데이터를 처리하므로 순서가 섞이지 않음
            appr_alg->addPoint((void *)(all_base_data + (long long)i * vecdim), (size_t)i);

            // 진행 상태 출력 (atomic 변수 사용)
            size_t current_progress = ++progress_counter;
            if (current_progress % report_every == 0) {
                #pragma omp critical 
                { // cout은 스레드 안전하지 않으므로 critical 블록으로 보호
                    cout << (float)current_progress * 100 / vecsize << " % processed, "
                        << "Mem: " << getCurrentRSS() / 1000000 << " Mb" << endl;
                }
            }
        }

        cout << "Build time: " << 1e-6 * stopw_full.getElapsedTimeMicro() << " seconds\n";
        appr_alg->saveIndex(path_index);
        
        // 할당한 메모리 해제
        delete[] all_base_data;
    }
    
    vector<std::priority_queue<std::pair<int, labeltype >>> answers;
    cout << "Parsing gt:\n";
    
    // gt_k 개의 이웃 중 k개를 사용해 정답셋 구성
    (vector<std::priority_queue<std::pair<int, labeltype >>>(qsize)).swap(answers);
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[gt_k * i + j]);
        }
    }
    cout << "Loaded gt\n";

    test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    
    delete[] massQA;
    delete[] massQ;
    delete[] mass;
    delete[] massb;
    delete appr_alg;

    return;
}