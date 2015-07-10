// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include "cuda_runtime_api.h"

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
//#include "commandline.h"

#ifdef WIN32
#undef min
#undef max
#endif

#include <limits>


using namespace cudpp_app;

cudaDeviceProp devProps;

int verifyStringSort(unsigned int *valuesSorted,
                     unsigned char* stringVals, size_t numElements,
                     int stringSize, unsigned char termC)
{
    int retval = 0;

    for(unsigned int i = 0; i < numElements-1; ++i)
    {
        unsigned int add1, add2;
        add1 = valuesSorted[i];
        add2 = valuesSorted[i+1];

        unsigned char c1, c2;

        do
        {
            c1 = (stringVals[add1]);
            c2 = (stringVals[add2]);


            add1++;
            add2++;

        }
        while(c1 == c2 && c1 != termC && c2 != termC &&
              add1 < stringSize && add2 < stringSize);

        if (c1 > c2)
        {
            printf("Error comparing index %d to %d (%d > %d) "
                   "(add1 %d add2 %d)\n",
                   i, i+1, c1, c2, valuesSorted[i], valuesSorted[i+1]);
            return 1;
        }

    }
    return retval;
}

int verifyPackedStringSort(unsigned int *valuesSorted,
                           unsigned int* packedStringVals, size_t numElements,
                           unsigned int stringSize, unsigned char termC)
{
    int retval = 0;

    for(unsigned int i = 0; i < numElements-1; ++i)
    {
        unsigned int add1, add2;
        add1 = valuesSorted[i];
        add2 = valuesSorted[i+1];

        unsigned int c1, c2;

        do
        {
            c1 = (packedStringVals[add1]);
            c2 = (packedStringVals[add2]);


            add1++;
            add2++;

        }
        while(c1 == c2 && (c1&255) != termC && (c2&255) != termC &&
              add1 < stringSize && add2 < stringSize);

        if(c1 > c2)
        {
            printf("Error comparing index %d to %d (%d > %d) "
                   "(add1 %d add2 %d)\n",
                   i, i+1, c1, c2, valuesSorted[i], valuesSorted[i+1]);
            return 1;
        }

    }
    return retval;
}

void stringSortTest(CUDPPHandle theCudpp, CUDPPConfiguration config,
                   size_t numElements,
                   testrigOptions testOptions, bool quiet)
{

	struct stat finfo;
    int fd = fopen("", O_RDONLY);
	fstat(fd, &finfo);

	char *data =(char*)mmap(0, finfo.st_size+1, PROT_READ|PROT_WRITE, MAP_POPULATE|MAP_PRIVATE, fd, 0);
	std::vector<int> offset;

	int i = 0;
	fileSize = finfo.st_size;
	while( i < fileSize)



    int retval = 0;
    srand(44);

    unsigned int  *h_valSend, *d_address, *h_valuesSorted;

    unsigned char *d_stringVals;

    unsigned char *stringVals;

    config.algorithm = CUDPP_SORT_STRING;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_FORWARD;


    unsigned int stringSize = numElements;

    h_valSend = (unsigned int*)malloc(numElements*sizeof(unsigned int));        // input val data
   
 
    h_valuesSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));

  
   
    stringVals = (unsigned char*) malloc(sizeof(unsigned char)*stringSize);    // input string data
  
   

      
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_address,
                              numElements*sizeof(unsigned int)));
							  
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_stringVals,
                              stringSize*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpy(d_stringVals,
                              stringVals,
                              stringSize*sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

    CUDPPHandle plan;
    CUDPPResult result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);



    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        cudppDestroyPlan(plan);
        return;
    }

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );



        if (!quiet)
        {

            printf("Running a string sort of %ld keys\n", numElements);
            fflush(stdout);
        }

        float totalTime = 0;

        for (int i = 0; i < testOptions.numIterations; i++)
        {
            CUDA_SAFE_CALL( cudaMemcpy(d_address, h_valSend,
                                       numElements * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice) );
            CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );


            cudppStringSort(plan, d_stringVals, d_address, 0, numElements,
                            stringSize);

            CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

            float time = 0;
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event,
                                                 stop_event));
            totalTime += time;
        }

        CUDA_CHECK_ERROR("teststringSort - cudppStringSort");

        // copy results

        CUDA_SAFE_CALL( cudaMemcpy(h_valuesSorted,
                                   d_address,
                                   numElements * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));

        CUDA_SAFE_CALL( cudaMemcpy(stringVals,
                                   d_stringVals,
                                   stringSize * sizeof(unsigned char),
                                   cudaMemcpyDeviceToHost));

        retval += verifyStringSort(h_valuesSorted,
                                   stringVals, numElements, stringSize, 0);

        //Verify that the keys make sense
        //TODO: Verify that all strings are in correct order using addresses

        if(!quiet)
        {
            printf("test %s\n", (retval == 0) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   totalTime / testOptions.numIterations);
        }
        else
        {
            printf("\t%0.4f\n",totalTime / testOptions.numIterations);
        }
    
    printf("\n");



    CUDA_CHECK_ERROR("after stringsort");

    result = cudppDestroyPlan(plan);


    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for StringSort\n");
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaFree(d_address);
    cudaFree(d_stringVals);


    free(h_valSend);
    free(h_valuesSorted);
    free(stringVals);
}


/**
 * testStringSort tests cudpp's merge sort
 * Possible command line arguments:
 * - -n=#, number of elements in sort
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppSort
 */
void testStringSort(int argc, const char **argv,
                   const CUDPPConfiguration *configPtr)
{
    int retval = 0;

    bool quiet = checkCommandLineFlag(argc, argv, "quiet");
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_STRING;
    config.datatype = CUDPP_UINT;



    // small GPUs are susceptible to running out of memory,
    // restrict the tests to only those where we have enough
    size_t freeMem, totalMem;
    CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    printf("freeMem: %d, totalMem: %d\n", int(freeMem), int(totalMem));
   

    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);

    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        return;
    }

	int numElements = 0;   // te be added

    stringSortTest(theCudpp, config, numElements,
                            testOptions, quiet);
    result = cudppDestroy(theCudpp);

}

int main(int argc, const char** argv)
{
   bool quiet = checkCommandLineFlag(argc, argv, "quiet");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    commandLineArg(dev, argc, argv, "device");
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    CUDA_SAFE_CALL( cudaSetDevice(dev) );

    CUDA_SAFE_CALL( cudaGetDeviceProperties(&devProps, dev) );
    if (!quiet)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %luB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, devProps.totalGlobalMem, (int)devProps.major,
               (int)devProps.minor, (int)devProps.clockRate);
        int runtimeVersion, driverVersion;
        CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtimeVersion));
        CUDA_SAFE_CALL(cudaDriverGetVersion(&driverVersion));
        printf("Driver API: %d; driver version: %d; runtime version: %d\n",
               CUDA_VERSION, driverVersion, runtimeVersion);
    }

    int computeVersion = devProps.major * 10 + devProps.minor;
    bool supportsDouble = (computeVersion >= 13);
    bool supports48KBInShared = (computeVersion >= 20);

    int retval = 0;

    if (argc == 1 || checkCommandLineFlag(argc, argv, "help"))
    {
        printf("Usage: \"cudpp_testrig -<flag> -<option>=<value>\"\n\n");
        printf("--- Global Flags ---\n");
        printf("all: Run all tests\n");
        printf("scan: Run scan test(s)\n");
        printf("stringsort: Run string sort test(s)\n");
        printf("--- Global Options ---\n");
        printf("iterations=<N>: Number of times to run each test\n");
        printf("n=<N>: Number of values to use in a single test\n");
        printf("r=<N>: Number of rows to scan (--multiscan only)\n\n");
        printf("--- Scan (Segmented and Unsegmented) Options ---\n");
        printf("backward: Run backward scans\n");
        printf("forward: Run forward scans (default)\n");
        printf("op=<OP>: Set scan operation to OP "
               "(OP=\"sum\", \"max\" \"min\" and \"multiply\"  currently. "
               "Default is sum)\n");
        printf("inclusive: Run inclusive scan (default)\n");
        printf("Exclusive: Run exclusive scan \n\n");
        printf("--- Radix Sort Options ---\n");
        printf("uint: Run radix sort on unsigned int keys (default)\n");
        printf("float: Run radix sort on float keys\n");
        printf("keyval: Run radix sort on key/value pairs (default)\n");
        printf("keysonly: Run radix sort on keys only\n");
        printf("forward: Run forward sorts (default)\n");
    }
	
	 bool runAll = checkCommandLineFlag(argc, argv, "all");
	// bool runScan = runAll || checkCommandLineFlag(argc, argv, "scan");
	 bool runStringSort = runAll || checkCommandLineFlag(argc, argv, "stringsort");
	 
	 bool hasopts = hasOptions(argc, argv);

    if (hasopts)
    {
        printf("has opts\n");
  //      if (runScan)      retval += testScan(argc, argv, NULL, false, devProps);
        if (runStringSort)
			testStringSort(argc, argv, NULL);
    }
    else
    {
        CUDPPConfiguration config;
        config.options = 0;

    //    if (runScan) {
//            config.algorithm = CUDPP_SCAN;
  //          retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble);
    //    }

        if(runStringSort) {
            config.algorithm = CUDPP_SORT_STRING;
            testStringSort(argc, argv, config);
          //  retval += testAllOptionsAndDatatypes(argc, argv, config, supportsDouble);
        }

       
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
