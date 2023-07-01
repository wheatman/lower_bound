OPT?=fast
VALGRIND?=0
INFO?=0
DEBUG?=0
SANITIZE?=0
DEBUG_SYMBOLS?=1
EXTRA_WARNINGS?=0
CYCLE_TIMER?=1


CFLAGS := -Wall -Wextra -O$(OPT)  -std=c++20 -DCYCLE_TIMER=$(CYCLE_TIMER) -march=native

ifeq ($(DEBUG_SYMBOLS),1)
CFLAGS += -g -gdwarf-4
endif



ifeq ($(EXTRA_WARNINGS),1)
CFLAGS += -Weverything
endif


ifeq ($(SANITIZE),1)
CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
endif

ifeq ($(INFO), 1) 
# CFLAGS +=  -Rpass-missed="(inline|loop*)" 
#CFLAGS += -Rpass="(inline|loop*)" -Rpass-missed="(inline|loop*)" -Rpass-analysis="(inline|loop*)" 
CFLAGS += -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.* 
endif

find: find.cpp utils.h
	$(CXX) $(CFLAGS)  -o find find.cpp

lower_bound: lower_bound.cpp utils.h
	$(CXX) $(CFLAGS)  -o lower_bound lower_bound.cpp