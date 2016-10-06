// Standard Library 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// For OpenMP
#include <omp.h>

using namespace std;

int main(int argc, char**argv) {
    if(argc>1)
        omp_set_num_threads(atoi(argv[1]));

    int i;     // Loop counter
	int count; // Number of successful trials
    int niter; // Total trials
    niter = 100000000;

    double x; // X coordinate
    double y; // Y coordinate
    double pi; // Estimate of Pi

    clock_t start_time;
    clock_t end_time;

    start_time = clock();
    #pragma omp parallel for private(i, x, y) reduction(+:count)
    for(i=0; i < niter; i++) {
       x = (double)rand()/RAND_MAX;
       y = (double)rand()/RAND_MAX;
       if(x*x + y*y < 1.0)
         count++;
    }
    end_time = clock();
    
    // Compute Pi
    pi = 4.0*(double)count/(double)niter;
    printf("# of trials= %d , estimate of pi is %g, time= %f \n",niter,pi, difftime(end_time, start_time));
    return 0;
}

