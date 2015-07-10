
OPT= -O3

NVFLAGS=$(OPT) -arch=sm_35
 
CC=g++
NVCC=nvcc 

OBJS=test_stringsort.o cudpp_plan.o cudpp.o cudpp_manager.o cudpp_maximal_launch.o cudpp_testrig_options.o stringsort_app.o scan_app.o 
//OBJS=stringsort_app.o cudpp_testrig_options.o cudpp_maximal_launch.o cudpp_manager.o cudpp.o cudpp_plan.o test_stringsort.o

.PHONY: main 

main: test_stringsort.o stringsort_app.o scan_app.o cudpp_plan.o cudpp.o cudpp_testrig_options.o cudpp_manager.o 
	$(NVCC) -ccbin=$(CC) $(NVFLAGS) -o main $^

%.o : %.cu
	$(NVCC) -ccbin=$(CC) $(NVFLAGS) -c $< -o $*.o 
%.o : %.cpp %.h
	$(NVCC) -ccbin=$(CC) $(CFLAGS) -c $< -o $*.o
.cpp.o :
	$(NVCC) -ccbin=$(CC) $(NVFLAGS) -c $< -o $*.o

clean:
	-rm main $(OBJS)
