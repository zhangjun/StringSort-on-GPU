// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * cudpp.cpp
 *
 * @brief Main library source file.  Implements wrappers for public
 * interface.
 *
 * Main library source file.  Implements wrappers for public
 * interface.  These wrappers call application-level operators.
 * As this grows we may decide to partition into multiple source
 * files.
 */

/**
 * \defgroup publicInterface CUDPP Public Interface
 * The CUDA public interface comprises the functions, structs, and enums
 * defined in cudpp.h.  Public interface functions call functions in the
 * \link cudpp_app Application-Level\endlink interface. The public
 * interface functions include Plan Interface functions and Algorithm
 * Interface functions.  Plan Interface functions are used for creating
 * CUDPP Plan objects that contain configuration details, intermediate
 * storage space, and in the case of cudppSparseMatrix(), data.  The
 * Algorithm Interface is the set of functions that do the real work
 * of CUDPP, such as cudppScan() and cudppSparseMatrixVectorMultiply().
 *
 * @{
 */

/** @name Algorithm Interface
 * @{
 */


#include "cudpp.h"
#include "cudpp_scan.h"
#include "cudpp_stringsort.h"
/**
 * @brief Performs a scan operation of numElements on its input in
 * GPU memory (d_in) and places the output in GPU memory
 * (d_out), with the scan parameters specified in the plan pointed to by
 * planHandle.

 * The input to a scan operation is an input array, a binary associative
 * operator (like + or max), and an identity element for that operator
 * (+'s identity is 0). The output of scan is the same size as its input.
 * Informally, the output at each element is the result of operator
 * applied to each input that comes before it. For instance, the
 * output of sum-scan at each element is the sum of all the input
 * elements before that input.
 *
 * More formally, for associative operator
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly,
 * <var>out<sub>i</sub></var> = <var>in<sub>0</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>1</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly ...
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>i-1</sub></var>.
 *
 * CUDPP supports "exclusive" and "inclusive" scans. For the ADD operator,
 * an exclusive scan computes the sum of all input elements before the
 * current element, while an inclusive scan computes the sum of all input
 * elements up to and including the current element.
 *
 * Before calling scan, create an internal plan using cudppPlan().
 *
 * After you are finished with the scan plan, clean up with cudppDestroyPlan().
 *
 * @param[in] planHandle Handle to plan for this scan
 * @param[out] d_out output of scan, in GPU memory
 * @param[in] d_in input to scan, in GPU memory
 * @param[in] numElements number of elements to scan
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, cudppDestroyPlan
 */
CUDPP_DLL
CUDPPResult cudppScan(const CUDPPHandle planHandle,
                      void              *d_out,
                      const void        *d_in,
                      size_t            numElements)
{
    CUDPPScanPlan *plan =
        (CUDPPScanPlan*)getPlanPtrFromHandle<CUDPPScanPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SCAN)
            return CUDPP_ERROR_INVALID_PLAN;

        cudppScanDispatch(d_out, d_in, numElements, 1, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Sorts strings. Keys are the first four characters of the string,
 * and values are the addresses where the strings reside in memory (stringVals)
 *
 * Takes as input an array of strings arranged as a char* array with
 * NULL terminating characters. This function will reformat this info
 * into keys (first four chars) values(pointers to string array
 * addresses) and aligned string value array.
 *
 *
 *
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[in] d_stringVals Original string input, no need for alignment or offsets.
 * @param[in] d_address Pointers (in order) to each strings starting location in the stringVals array
 * @param[in] termC Termination character used to separate strings
 * @param[in] numElements number of strings
 * @param[in] stringArrayLength Length in uint of the size of all strings
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppStringSort(const CUDPPHandle planHandle,
                      unsigned char     *d_stringVals,
                                          unsigned int      *d_address,
                                          unsigned char     termC,
                      size_t            numElements,
                      size_t            stringArrayLength)
{
    CUDPPStringSortPlan *plan =
        (CUDPPStringSortPlan*)getPlanPtrFromHandle<CUDPPStringSortPlan>(planHandle);      // reinterpret_cast, CUDPPHandle -> CUDPPStringSortPlan


    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SORT_STRING)
            return CUDPP_ERROR_INVALID_PLAN;

        unsigned int* packedStringVals;
        unsigned int *packedStringLength = (unsigned int*)malloc(sizeof(unsigned int));;

        calculateAlignedOffsets(d_address, plan->m_numSpaces, d_stringVals, termC, numElements, stringArrayLength);        // caculate the padding length to align, store in d_numSpaces
        cudppScanDispatch(plan->m_spaceScan, plan->m_numSpaces, numElements+1, 1, plan->m_scanPlan);                       // scanArrayRecursive prefix sum
        dotAdd(d_address, plan->m_spaceScan, plan->m_packedAddress, numElements+1, stringArrayLength);                     // add m_spaceScan to d_address

        cudaMemcpy(packedStringLength, (plan->m_packedAddress)+numElements, sizeof(unsigned int), cudaMemcpyDeviceToHost); // the last one in packedAddress
        cudaMemcpy(plan->m_packedAddressRef, plan->m_packedAddress, sizeof(unsigned int)*numElements, cudaMemcpyDeviceToDevice);
        cudaMemcpy(plan->m_addressRef, d_address, sizeof(unsigned int)*numElements, cudaMemcpyDeviceToDevice);

        //system("PAUSE");
        cudaMalloc((void**)&packedStringVals, sizeof(unsigned int)*packedStringLength[0]);        // allocate memory for packedStringVals

        packStrings(packedStringVals, d_stringVals, plan->m_keys, plan->m_packedAddress, d_address, numElements, stringArrayLength, termC);     // d_stringVals -> packedStrings  and create m_keys

        cudppStringSortDispatch(plan->m_keys, plan->m_packedAddress, packedStringVals, numElements, packedStringLength[0], termC, plan);
        unpackStrings(plan->m_packedAddress, plan->m_packedAddressRef, d_address, plan->m_addressRef, numElements);


        free(packedStringLength);
        cudaFree(packedStringVals);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/** @} */ // end Algorithm Interface
/** @} */ // end of publicInterface group

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
