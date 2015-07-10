// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp.h"
#include "cudpp_manager.h"
#include "cudpp_scan.h"
#include "cudpp_stringsort.h"
#include "cuda_util.h"
#include <cuda_runtime_api.h>

#include <assert.h>

CUDPPResult validateOptions(CUDPPConfiguration config, size_t numElements, size_t numRows, size_t /*rowPitch*/)
{
    CUDPPResult ret = CUDPP_SUCCESS;
    if ((config.options & CUDPP_OPTION_BACKWARD) && (config.options & CUDPP_OPTION_FORWARD))
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    if ((config.options & CUDPP_OPTION_EXCLUSIVE) && (config.options & CUDPP_OPTION_INCLUSIVE))
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION;


    return ret;
}

/** @addtogroup publicInterface
  * @{
  */

/** @name Plan Interface
 * @{
 */


/** @brief Create a CUDPP plan 
  * 
  * A plan is a data structure containing state and intermediate storage space
  * that CUDPP uses to execute algorithms on data.  A plan is created by 
  * passing to cudppPlan() a CUDPPConfiguration that specifies the algorithm,
  * operator, datatype, and options.  The size of the data must also be passed
  * to cudppPlan(), in the \a numElements, \a numRows, and \a rowPitch 
  * arguments.  These sizes are used to allocate internal storage space at the
  * time the plan is created.  The CUDPP planner may use the sizes, options,
  * and information about the present hardware to choose optimal settings.
  *
  * Note that \a numElements is the maximum size of the array to be processed
  * with this plan.  That means that a plan may be re-used to process (for 
  * example, to sort or scan) smaller arrays.  
  * 
  * @param[out] planHandle A pointer to an opaque handle to the internal plan
  * @param[in]  cudppHandle A handle to an instance of the CUDPP library used for resource management
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numElements The maximum number of elements to be processed
  * @param[in]  numRows The number of rows (for 2D operations) to be processed
  * @param[in]  rowPitch The pitch of the rows of input data, in elements
  * @returns CUDPPResult indicating success or error condition
  */
CUDPP_DLL
CUDPPResult cudppPlan(const CUDPPHandle  cudppHandle,
                      CUDPPHandle        *planHandle,
                      CUDPPConfiguration config, 
                      size_t             numElements, 
                      size_t             numRows, 
                      size_t             rowPitch)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPPlan *plan;
    CUDPPManager *mgr = CUDPPManager::getManagerFromHandle(cudppHandle);

    result = validateOptions(config, numElements, numRows, rowPitch);
    if (result != CUDPP_SUCCESS)
    {
        *planHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    switch (config.algorithm)
    {
    case CUDPP_SCAN:
        {
            plan = new CUDPPScanPlan(mgr, config, numElements, numRows, rowPitch);
            break;
        }
    case CUDPP_SORT_STRING:
        {
            plan = new CUDPPStringSortPlan(mgr, config, numElements, rowPitch);
            break;
        }	
    default:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION; 
        break;
    }

    if (!plan)
        return CUDPP_ERROR_UNKNOWN;
    else
    {
        *planHandle = plan->getHandle();
        return CUDPP_SUCCESS;
    }
}

/** @brief Destroy a CUDPP Plan
  *
  * Deletes the plan referred to by \a planHandle and all associated internal
  * storage.
  * 
  * @param[in] planHandle The CUDPPHandle to the plan to be destroyed
  * @returns CUDPPResult indicating success or error condition
  */
CUDPP_DLL
CUDPPResult cudppDestroyPlan(CUDPPHandle planHandle)
{
    if (planHandle == CUDPP_INVALID_HANDLE)
        return CUDPP_ERROR_INVALID_HANDLE;

    CUDPPPlan* plan = getPlanPtrFromHandle<CUDPPPlan>(planHandle);

    switch (plan->m_config.algorithm)
    {
    case CUDPP_SCAN:
        {
            delete static_cast<CUDPPScanPlan*>(plan);
            break;
        }
    case CUDPP_SORT_STRING:
        {
            delete static_cast<CUDPPStringSortPlan*>(plan);
            break;
        }	
    default:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION; 
        break;
    }

    plan = 0;
    return CUDPP_SUCCESS;
}

CUDPPPlan::CUDPPPlan(CUDPPManager *mgr,
                     CUDPPConfiguration config, 
                     size_t numElements, 
                     size_t numRows, 
                     size_t rowPitch)
: m_config(config),
  m_numElements(numElements),
  m_numRows(numRows),
  m_rowPitch(rowPitch),
  m_planManager(mgr)
{
}

/** @brief Scan Plan constructor
* 
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying algorithm and options
* @param[in]  numElements The maximum number of elements to be scanned
* @param[in]  numRows The maximum number of rows (for 2D operations) to be scanned
* @param[in]  rowPitch The pitch of the rows of input data, in elements
*/
CUDPPScanPlan::CUDPPScanPlan(CUDPPManager *mgr,
                             CUDPPConfiguration config, 
                             size_t numElements, 
                             size_t numRows, 
                             size_t rowPitch)
: CUDPPPlan(mgr, config, numElements, numRows, rowPitch),
  m_blockSums(0),
  m_rowPitches(0),
  m_numEltsAllocated(0),
  m_numRowsAllocated(0),
  m_numLevelsAllocated(0)
{
    allocScanStorage(this);
}

/** @brief CUDPP scan plan destructor */
CUDPPScanPlan::~CUDPPScanPlan()
{
    freeScanStorage(this);
}

/** @brief String Sort Plan consturctor
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be sorted
* @param[in]  stringArrayLength The length of our input string (in uint)
*/
CUDPPStringSortPlan::CUDPPStringSortPlan(CUDPPManager *mgr,
										 CUDPPConfiguration config,
										 size_t numElements, 
										 size_t stringArrayLength)
: CUDPPPlan(mgr, config, numElements, stringArrayLength, 0)
{ 
	m_subPartitions = 4;
	m_swapPoint = 64;
	
	CUDPPConfiguration scanConfig = 
    { 
      CUDPP_SCAN, 
      CUDPP_ADD, 
      CUDPP_UINT, 
      CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE 
    };    
	

	m_scanPlan = new CUDPPScanPlan(mgr, scanConfig, numElements+1, 1, 0);	
	m_numElements = numElements;
	allocStringSortStorage(this);	
}

/** @brief String sort plan destructor */
CUDPPStringSortPlan::~CUDPPStringSortPlan()
{
    freeStringSortStorage(this);
}

