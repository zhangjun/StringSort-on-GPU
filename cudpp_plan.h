// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef __CUDPP_PLAN_H__
#define __CUDPP_PLAN_H__

typedef void* KernelPointer;
class CUDPPPlan;
class CUDPPManager;

#include "cudpp.h"

//! @internal Convert an opaque handle to a pointer to a plan
template <typename T>
T* getPlanPtrFromHandle(CUDPPHandle handle)
{
    return reinterpret_cast<T*>(handle);
}


/** @brief Base class for CUDPP Plan data structures
  *
  * CUDPPPlan and its subclasses provide the internal (i.e. not visible to the
  * library user) infrastructure for planning algorithm execution.  They 
  * own intermediate storage for CUDPP algorithms as well as, in some cases,
  * information about optimal execution configuration for the present hardware.
  * 
  */
class CUDPPPlan
{
public:
    CUDPPPlan(CUDPPManager *mgr, CUDPPConfiguration config, 
              size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPPlan() {}

    // Note anything passed to functions compiled by NVCC must be public
    CUDPPConfiguration m_config;        //!< @internal Options structure
    size_t             m_numElements;   //!< @internal Maximum number of input elements
    size_t             m_numRows;       //!< @internal Maximum number of input rows
    size_t             m_rowPitch;      //!< @internal Pitch of input rows in elements
    CUDPPManager      *m_planManager;  //!< @internal pointer to the manager of this plan
   
    //! @internal Convert this pointer to an opaque handle
    //! @returns Handle to a CUDPP plan
    CUDPPHandle getHandle()
    {
        return reinterpret_cast<CUDPPHandle>(this);
    }
};

/** @brief Plan class for scan algorithm
  *
  */
class CUDPPScanPlan : public CUDPPPlan
{
public:
    CUDPPScanPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPScanPlan();

    void  **m_blockSums;          //!< @internal Intermediate block sums array
    size_t *m_rowPitches;         //!< @internal Pitch of each row in elements (for cudppMultiScan())
    size_t  m_numEltsAllocated;   //!< @internal Number of elements allocated (maximum scan size)
    size_t  m_numRowsAllocated;   //!< @internal Number of rows allocated (for cudppMultiScan())
    size_t  m_numLevelsAllocated; //!< @internal Number of levels allocaed (in _scanBlockSums)
};

/** @brief Plan class for stringsort algorithm
*
*/

class CUDPPStringSortPlan : public CUDPPPlan
{
public:
    CUDPPStringSortPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements, size_t stringArrayLength);
    virtual ~CUDPPStringSortPlan();

    unsigned int m_stringArrayLength;

	CUDPPScanPlan *m_scanPlan;
	unsigned int m_numElements;
	unsigned int *m_keys;
        unsigned int *m_tempKeys;
        unsigned int *m_tempAddress;
	unsigned int *m_packedAddress;
	unsigned int *m_packedAddressRef;
	unsigned int *m_addressRef;
	unsigned int *m_numSpaces;
	unsigned int *m_spaceScan;

	unsigned int m_subPartitions, m_swapPoint;
	unsigned int *m_partitionSizeA, *m_partitionSizeB, *m_partitionStartA, *m_partitionStartB;


	
};



#endif // __CUDPP_PLAN_H__
